from collections import OrderedDict
from os import pread
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Transformer
from wandb import Image
from model_layers import TransformerEncoder, TransformerEncoderLayer
import librosa
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from collections import namedtuple
from typing import NamedTuple

ForwardResult = namedtuple(
    "ForwardResult",
    [
        "pred_spec",
        "pred_time",
        "trues",
        "true_spec",
        "decoder_output",
        "filtered",
        "unfiltered",
        "teacher_forcing",
        "freq_importance_weights",
        "lambdas",
    ],
)
StepOutputs = NamedTuple(
    "EpochOutputs", [("ForwardResult", list[ForwardResult]), ("loss", list)]
)

torch.set_float32_matmul_precision("medium")


class SSF(pl.LightningModule):
    def __init__(self, args, data_args):
        super(SSF, self).__init__()
        self.args = args
        self.data_args = args

        self.train_step_outputs = StepOutputs([], [])
        self.validation_step_outputs = StepOutputs([], [])
        self.test_step_outputs = StepOutputs([], [])

        self.save_hyperparameters()
        self.generator = np.random.default_rng(2023)

        self.features = (
            self.args.nfft // 2 + 1
        ) * 2  # times 2 because of real and imag part

        self.filter = nn.Conv1d(
            in_channels=self.features,
            out_channels=self.features,
            groups=self.features,
            kernel_size=self.args.filterk,
            padding=self.args.filterk // 2,
        )
        self.mean = nn.Conv1d(
            in_channels=self.features,
            out_channels=self.features,
            kernel_size=self.args.mean_kernel_size,
            groups=2,
            padding=self.args.mean_kernel_size // 2,
        )
        self.var = nn.Conv1d(
            in_channels=self.features,
            out_channels=self.features,
            kernel_size=self.args.var_kernel_size,
            groups=2,
            padding=self.args.var_kernel_size // 2,
        )

        self.lambda_parameter = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=self.features),
            nn.ReLU(),
            nn.Linear(in_features=self.features, out_features=self.features),
            nn.ReLU(),
            nn.Linear(in_features=self.features, out_features=self.features),
            nn.Sigmoid(),
        )

        self.feature_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "Linear1",
                        nn.Linear(
                            in_features=self.args.inchannels_conf,
                            out_features=self.args.inchannels_conf * 2,
                        ),
                    ),
                    ("Activation1", nn.ReLU()),
                    (
                        "Linear2",
                        nn.Linear(
                            in_features=self.args.inchannels_conf * 2,
                            out_features=self.features,
                        ),
                    ),
                ]
            )
        )

        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.features,
            nhead=self.args.heads,
            nhead_low=self.args.heads_low,
            dropout=self.args.dropout,
            kernel_low=args.kernel_low,
            kernel_high=args.kernel_high,
            dilation_low=args.dilation_low,
            dilation_high=args.dilation_high,
            avgPoolK=args.avgPoolK,
            # avgPoolS=args.avgPoolS,
            time_compression=args.time_compression,
            # attn='full'
        )

        encoder = TransformerEncoder(self.encoder_layer, num_layers=self.args.nencoder)

        self.model = Transformer(
            d_model=self.features,
            nhead=self.args.heads,
            custom_encoder=encoder,
            norm_first=True,
        )  # d_model must be divisible by nhead

        fix_series = list(
            pd.read_excel("DATASETS/ton_10_kmh_50_sano.xlsx", header=None)[1]
        )
        fix_series.extend(1000 * [0])
        self.reference_signals = [
            list(pd.read_excel("DATASETS/ton_10_kmh_20_crack_50.xlsx", header=None)[1]),
            list(pd.read_excel("DATASETS/ton_10_kmh_20_sano.xlsx", header=None)[1]),
            list(pd.read_excel("DATASETS/ton_10_kmh_50_crack_50.xlsx", header=None)[1]),
            fix_series,
        ]

        self.len_sequence = (
            2000 // self.args.hop
            if 2000 % self.args.hop == 0
            else 2000 // self.args.hop + 1
        )
        self.len_source = int(self.len_sequence * (1 - 0.1))
        self.len_forecast = self.len_sequence - self.len_source


        self.ref_embeddings = nn.Linear(
            in_features=2000, out_features=self.features // 2
        )

        self.freq_importance = nn.Sequential(
            nn.Linear(5, 8),  # Input layer
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Dropout(p=0.3),  # Dropout layer
            nn.Linear(8, 16),  # First hidden layer
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 32),  # Second hidden layer
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(p=0.3),  # Dropout layer
            nn.Linear(32, self.features // 2),  # Output layer
            nn.Softmax(dim=1),
        )

    @staticmethod
    def pos_encoding(x):
        def sin_fun(pos, i, d):
            return torch.sin(torch.tensor(pos / 10000 ** (2 * i / d)))

        def cos_fun(pos, i, d):
            return torch.cos(torch.tensor(pos / 10000 ** (2 * i / d)))

        B, d, l = x.shape

        for i in range(d):
            for pos in range(l):
                pe = sin_fun if pos % 2 == 0 else cos_fun
                x[:, i, pos] = x[:, i, pos] + pe(pos, i, d)

        return x

    def _autorregresive_model(self, source, target, B, test_teacher_forcing_every=None):
        last_position_memory_recalculated = 0
        memory = self.model.encoder(source)
        out_decoder = torch.zeros((1, B, self.features), device=self.device)
        out_decoder[0, :, :] = source[-1, :, :]
        for forecasted_position in range(self.len_sequence - self.len_source):
            recalculate_memory = False

            if (forecasted_position % self.args.recalculate_memory_every == 0) and (
                forecasted_position != 0
            ):
                recalculate_memory = True

            if recalculate_memory:
                memory = self.model.encoder(
                    torch.cat([source, out_decoder[1:, :, :]], dim=0)
                )
                last_position_memory_recalculated += self.args.recalculate_memory_every

            target_tokens = out_decoder[
                last_position_memory_recalculated : forecasted_position + 1, :, :
            ]

            if (
                (test_teacher_forcing_every is not None)
                and (forecasted_position % test_teacher_forcing_every == 0)
                and (forecasted_position != 0)
            ):
                future_point = target[forecasted_position, :, :].unsqueeze(0)
            else:
                future_point = self.model.decoder(target_tokens, memory)[
                    -1, :, :
                ].unsqueeze(0)

            out_decoder = torch.cat([out_decoder, future_point], dim=0)
        future_point = out_decoder[1:, :, :]

        return future_point

    def _compute_params_and_sample(self, future_point):
        # Compute mean and variance of the forecasted spectrogram and then sample
        mean = self.mean(future_point.permute(1, 2, 0)).permute(2, 0, 1)
        mean = nn.functional.tanh(mean) * self.args.feature_range[1]
        real = mean[:, :, : mean.shape[2] // 2]
        imag = mean[:, :, mean.shape[2] // 2 :]
        mean_complex = torch.complex(real, imag).permute(1, 2, 0)

        var_complex, lambdas = self._compute_variance_and_sample(future_point)

        noise = torch.randn_like(mean_complex) * torch.sqrt(var_complex)
        pred_spec = mean_complex + noise  # *self.noise_param
        return pred_spec, lambdas

    def _compute_variance_and_sample(self, future_point):
        # Compute mean and variance of the forecasted spectrogram and then sample
        lambda_parameter = 1 + 1 / (
            0.01 + self.lambda_parameter(future_point) ** 2
        )  # torch.exp()

        real = lambda_parameter[:, :, : lambda_parameter.shape[2] // 2]
        imag = lambda_parameter[:, :, lambda_parameter.shape[2] // 2 :]
        lambda_parameter_complex = torch.complex(real, imag).permute(1, 2, 0)

        # Inversion method (sampling)
        u = torch.rand_like(lambda_parameter_complex, device=self.device)
        sample = -1 / lambda_parameter_complex * torch.log(u)

        return sample, lambda_parameter

    def _merge_output_with_reference(
        self, feat, window, future_point, trues, filtered, unfiltered, teacher_forcing
    ):
        # Reference signal
        reference_signal = self._process_reference_signal(feat, self.generator, window)

        # Merge the vanilla decoder output with the reference signal and the features
        feature_embeddings = self.feature_linear(feat)
        ref_emb = self.ref_embeddings(reference_signal)
        future_point = (
            future_point + feature_embeddings
        )

        freq_importance = self.freq_importance(feat + ref_emb)
        freq_importance_weights = torch.div(
            freq_importance.permute(1, 0), freq_importance.max(dim=1).values
        ).permute(1, 0)

        future_point = future_point.reshape(
            future_point.shape[1], future_point.shape[0], 2, future_point.shape[2] // 2
        )
        future_point = future_point * freq_importance.reshape(
            freq_importance_weights.shape[0], 1, 1, freq_importance_weights.shape[1]
        ).repeat(1, future_point.shape[1], 2, 1)
        future_point = future_point.reshape(
            future_point.shape[1], future_point.shape[0], -1
        )
        pred_spec, lambdas = self._compute_params_and_sample(future_point)

        # Transform the forecasted spectrogram back into the time domain
        pred_time = torch.istft(
            pred_spec,
            n_fft=self.args.nfft,
            hop_length=self.args.hop,
            window=window,
            return_complex=False,
            onesided=True,
            center=True,
            length=trues.shape[0],
        )
        assert (
            pred_time.shape[0] == trues.shape[1]
        ), f"Shapes: {pred_time.shape[0]} /  {trues.shape[1]}"

        true_spec = torch.stft(
            trues.T,
            n_fft=self.args.nfft,
            hop_length=self.args.hop,
            window=window,
            return_complex=True,
            onesided=True,
        )
        if pred_spec.shape == true_spec[:, :, :-1].shape:
            true_spec = true_spec[:, :, :-1]
        assert pred_spec.shape == true_spec.shape, str(pred_spec.shape) + str(
            true_spec.shape
        )
        return (
            pred_spec,
            pred_time.permute(1, 0),
            trues,
            true_spec,
            future_point,
            filtered,
            unfiltered,
            teacher_forcing,
            freq_importance_weights,
            lambdas,
        )

    def _get_attention_mask(self, batch_size, seq_len):
        # Create tensor with shape (batch_size, seq_len, seq_len) initialized with ones
        attention_mask = torch.ones(batch_size * self.args.heads, seq_len, seq_len)

        # Set lower triangular part of attention mask to 0
        attention_mask = torch.triu(attention_mask, diagonal=1)
        new_attn_mask = attention_mask * -1e7 + 1

        """After applying softmax:
         [0., 0., 0.,  ..., 0., 0., 0.],
         [1., 0., 0.,  ..., 0., 0., 0.],
         [1., 1., 0.,  ..., 0., 0., 0.],
         ...,
         [1., 1., 1.,  ..., 0., 0., 0.],
         [1., 1., 1.,  ..., 1., 0., 0.],
         [1., 1., 1.,  ..., 1., 1., 0.]"""

        return new_attn_mask

    def _process_reference_signal(self, feat, generator, window):
        # output is a L=2000 signal to match the other signals. Also, the frequency channels are condensed to just one
        reference_signal = []
        for f in feat:
            if f[0]:
                if f[1]:
                    reference_signal.append(
                        self.reference_signals[int(generator.integers(0, 2, size=1))][
                            :-1
                        ]
                    )  # 4t 20kmh - ahora 10t
                else:
                    reference_signal.append(
                        self.reference_signals[int(generator.integers(2, 4, size=1))][
                            :-1
                        ]
                    )  # 4t 50kmh - ahora 10t
            if not f[0]:
                if f[1]:
                    reference_signal.append(
                        self.reference_signals[int(generator.integers(0, 2, size=1))][
                            :-1
                        ]
                    )  # 10t 20kmh
                else:
                    reference_signal.append(
                        self.reference_signals[int(generator.integers(2, 4, size=1))][
                            :-1
                        ]
                    )  # 10t 50kmh

        reference_signal = torch.abs(
            torch.fft.fft(
                torch.tensor(reference_signal, dtype=torch.float32, device=self.device)[
                    :, 0::4
                ]
            )
        )
        return reference_signal  # B x 2000 x 1

    def forward(
        self,
        x: torch.Tensor,
        feat: torch.Tensor,
        teacher_forcing=True,
        results_phase=False,
    ):
        if not teacher_forcing:
            test = True
        else:
            test = False

        x = x.to(self.device)
        feat = feat.to(self.device)
        B = x.shape[1]

        chance = self.generator.choice(
            np.array([0, 1]), size=1, p=np.array([1 - self.args.p, self.args.p])
        )
        if not test and bool(chance) and not results_phase:
            teacher_forcing = False

        if test:
            self.model.eval()
            self.filter.eval()

        # Compute the complex-valued spectrogram
        window = torch.hann_window(self.args.nfft).to(self.device)
        x_complex = torch.stft(
            x.T,
            n_fft=self.args.nfft,
            hop_length=self.args.hop,
            window=window,
            return_complex=True,
            onesided=True,
        )

        # POSITIONAL ENCODING
        if self.args.pe == 1:
            x_complex = self.pos_encoding(x_complex)

        # Compute the magnitude and phase components of the spectrogram
        x_real = x_complex.real.float()
        x_imag = x_complex.imag.float()

        # Stack the magnitude and phase components
        x_input = torch.stack([x_real, x_imag], dim=3).permute(2, 0, 1, 3)
        unfiltered = (
            x_input.reshape(x_input.shape[0], x_input.shape[1], -1).clone().detach()
        )

        # Filter the spectrogram
        x_input = self.filter(unfiltered.permute(1, 2, 0)).permute(2, 0, 1)
        filtered = x_input.clone().detach()
        assert x_input.shape[2] == self.features

        # Split the signals
        source = x_input[: self.len_source, :, :]
        target = x_input[self.len_source :, :, :]
        trues = x[self.len_source * self.args.hop :, :]

        # Calculate future points of the signal
        if teacher_forcing and not test:
            attention_mask = self._get_attention_mask(
                batch_size=B, seq_len=target.shape[0]
            )
            attention_mask = attention_mask.to(self.device)
            future_point = self.model(
                source,
                torch.cat((source[-1, :, :].unsqueeze(0), target[:-1, :, :]), 0),
                tgt_mask=attention_mask,
            )
            return self._merge_output_with_reference(
                feat, window, future_point, trues, filtered, unfiltered, teacher_forcing
            )

        if not teacher_forcing and test:
            with torch.no_grad():
                future_point = self._autorregresive_model(
                    source,
                    target,
                    B,
                    test_teacher_forcing_every=self.args.test_teacher_forcing_every,
                )
                return self._merge_output_with_reference(
                    feat,
                    window,
                    future_point,
                    trues,
                    filtered,
                    unfiltered,
                    teacher_forcing,
                )

        if not teacher_forcing and not test:
            future_point = self._autorregresive_model(source, target, B)
            return self._merge_output_with_reference(
                feat, window, future_point, trues, filtered, unfiltered, teacher_forcing
            )

    #### PLOT FUNCTION
    def plot_signals(
        self,
        pred,
        trues,
        teacher_forcing,
        name: str = "Train",
        target=None,
        row=1,
        col=1,
        mean=None,
        var=None,
    ):

        # Convert tensors to numpy arrays
        pred_np = pred.cpu().detach().numpy()[:, 0]
        true_np = trues.cpu().detach().numpy()[:, 0]

        # Compute STFT using PyTorch
        n_fft = self.args.nfft
        hop_length = self.args.hop
        window = torch.hann_window(n_fft)

        pred_stft = torch.stft(torch.tensor(pred_np), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, onesided=True)
        true_stft = torch.stft(torch.tensor(true_np), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, onesided=True)

        pred_spectrogram = torch.abs(pred_stft).numpy()
        true_spectrogram = torch.abs(true_stft).numpy()

        # Normalize the color scale
        vmin = min(pred_spectrogram.min(), true_spectrogram.min())
        vmax = max(pred_spectrogram.max(), true_spectrogram.max())

        # Calculate MSE
        mse = np.mean((pred_np - true_np) ** 2)

        # Time Domain Plot
        fig_time, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(true_np, label="Ground Truth", color='blue')
        ax1.plot(pred_np, label="Prediction", color='orange')
        ax1.set_title("Time Domain")
        ax1.set_xlabel("Time [samples]")
        ax1.set_ylabel("Amplitude")
        ax1.legend(loc='upper right')
        ax1.set_title(f"Time Domain\nMSE: {mse:.4f}", fontsize=14)

        plt.savefig(f"./results/{self.args.name_folder}/{name}_time.png")
        self.logger.experiment.log({f"{name}/Time Domain": fig_time})
        self.logger.log_image(f"{name}/Time Domain (png)", [fig_time])
        plt.close(fig_time)

        # Spectrograms Plotting
        fig_spec, axes = plt.subplots(1, 2, figsize=(18, 5))

        # Predicted Spectrogram
        im1 = axes[0].pcolormesh(pred_spectrogram, shading='gouraud', vmin=vmin, vmax=vmax)
        axes[0].set_title("Predicted Spectrogram")
        axes[0].set_xlabel("Time [frames]")
        axes[0].set_ylabel("Frequency [bins]")

        # True Spectrogram
        im2 = axes[1].pcolormesh(true_spectrogram, shading='gouraud', vmin=vmin, vmax=vmax)
        axes[1].set_title("True Spectrogram")
        axes[1].set_xlabel("Time [frames]")
        axes[1].set_ylabel("Frequency [bins]")
        cbar = fig_spec.colorbar(im2, ax=axes[1])
        cbar.set_label('Magnitude')

        fig_spec.suptitle(name, fontsize=16)
        plt.savefig(f"./results/{self.args.name_folder}/{name}_spectrograms.png")
        self.logger.log_image(f"{name}/Spectrograms", [fig_spec])
        plt.close(fig_spec)

        # Difference Spectrogram
        fig_diff, ax_diff = plt.subplots(figsize=(6, 5))
        diff_spectrogram = np.abs(true_spectrogram - pred_spectrogram)
        im_diff = ax_diff.pcolormesh(diff_spectrogram, shading='gouraud', cmap='gist_heat')
        ax_diff.set_title("Difference Spectrogram")
        ax_diff.set_xlabel("Time [frames]")
        ax_diff.set_ylabel("Frequency [bins]")
        cbar_diff = fig_diff.colorbar(im_diff, ax=ax_diff)
        cbar_diff.set_label('Magnitude Difference')
        plt.savefig(f"./results/{self.args.name_folder}/{name}_diff.png")
        self.logger.log_image(f"{name}/Difference Spectrogram", [fig_diff])
        plt.close(fig_diff)   

        if mean is not None and var is not None:
            fig = make_subplots(rows=row, cols=col)
            x = np.linspace(0, 199, 200)

            for i, perm in enumerate([[1, 1], [1, 2], [2, 1], [2, 2]]):
                mean_i = mean[:200, i, 0].detach().cpu().numpy()
                var_i = var[:200, i, 0].detach().cpu().numpy()

                fig.add_trace(
                    go.Scatter(
                        name="Expected prediction",
                        x=x,
                        y=mean_i,
                        mode="lines",
                        line=dict(color="rgb(255,164,0)"),
                    ),
                    row=perm[0],
                    col=perm[1],
                )
                fig.add_trace(
                    go.Scatter(
                        name="True signal",
                        x=x,
                        y=trues.cpu().detach().numpy()[:200, i],
                        mode="lines",
                        line=dict(color="rgb(31, 119, 180)"),
                    ),
                    row=perm[0],
                    col=perm[1],
                )
                fig.add_trace(
                    go.Scatter(
                        name="Upper Bound",
                        x=x,
                        y=mean_i + 2 * var_i,
                        mode="lines",
                        marker=dict(color="#444"),
                        line=dict(width=0.8),
                        showlegend=False,
                    ),
                    row=perm[0],
                    col=perm[1],
                )
                fig.add_trace(
                    go.Scatter(
                        name="Lower Bound",
                        x=x,
                        y=mean_i - 2 * var_i,
                        marker=dict(color="#444"),
                        line=dict(width=0.8),
                        mode="lines",
                        fillcolor="rgba(68, 68, 68, 0.3)",
                        fill="tonexty",
                        showlegend=False,
                    ),
                    row=perm[0],
                    col=perm[1],
                )

            fig.update_layout(
                height=600,
                width=800,
                hovermode="x",
                title_text="Mean prediction and 95% CI",
                showlegend=False,
            )
            self.logger.experiment.log({f"{name}/95% confidence intervals": fig})
            plt.savefig(f"./results/{self.args.name_folder}/{name}_CI.png")

    def logResults(
        self,
        decoder_output,
        prediction_spec,
        true,
        filtered,
        unfiltered,
        teacher_forcing,
        freq_importance_weights,
        lambdas,
        title,
    ):
        i = 0  # np.random.randint(0, prediction.shape[0], size=1)

        self.logger.experiment.log({"Frequency importance weights": freq_importance_weights})
        self.logger.experiment.log({"Lambdas": lambdas})

        if not teacher_forcing:
            teacher_forcing = ""
        else:
            teacher_forcing = ""

        # Log the decoder output
        real = decoder_output[:, :, : decoder_output.shape[2] // 2]
        imag = decoder_output[:, :, decoder_output.shape[2] // 2 :]
        decoder_output_complex = torch.complex(real, imag)
        decoder_output_amp = np.abs(decoder_output_complex[:, i, :].squeeze(1).cpu().detach().numpy()).T

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.pcolormesh(decoder_output_amp, shading='gouraud')
        ax.set_title("Decoder Output")
        ax.set_xlabel("Time [frames]")
        ax.set_ylabel("Frequency [bins]")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Magnitude")
        self.logger.log_image(f"{title}/decoder output", [fig])
        plt.close()

        # Log the un/filtered signal
        real = unfiltered[:, i, : unfiltered.shape[2] // 2]
        imag = unfiltered[:, i, unfiltered.shape[2] // 2 :]
        unfiltered_complex = torch.complex(real, imag).squeeze(1).cpu().detach().numpy()

        real = filtered[:, i, : filtered.shape[2] // 2]
        imag = filtered[:, i, filtered.shape[2] // 2 :]
        filtered_complex = torch.complex(real, imag).squeeze(1).cpu().detach().numpy()

        v_min, v_max = min(np.min(np.abs(filtered_complex)), np.min(np.abs(unfiltered_complex))), max(np.max(np.abs(filtered_complex)), np.max(np.abs(unfiltered_complex)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        im1 = axes[0].pcolormesh(np.abs(filtered_complex).T, shading='gouraud', vmax=v_max, vmin=v_min)
        axes[0].set_title("Filtered")
        axes[0].set_xlabel("Time [frames]")
        axes[0].set_ylabel("Frequency [bins]")

        im2 = axes[1].pcolormesh(np.abs(unfiltered_complex).T, shading='gouraud', vmax=v_max, vmin=v_min)
        axes[1].set_title("Unfiltered")
        axes[1].set_xlabel("Time [frames]")
        axes[1].set_ylabel("Frequency [bins]")

        fig.colorbar(im2, ax=axes.ravel().tolist()).set_label("Magnitude")
        self.logger.log_image(f"{title}/filter on the spectrogram", [fig])
        plt.close()



    def log_decomposition(self, mode, pred, true):
        # Convert predicted series to DataFrame
        series_pred = pred.detach().cpu().numpy()[:, 0]
        date_range = pd.date_range(start='2023-10-22', periods=len(series_pred), freq='M')
        df_pred = pd.DataFrame({'value': series_pred}, index=date_range)
        result_pred = seasonal_decompose(df_pred['value'], model='additive')

        # Convert true series to DataFrame
        series_true = true.detach().cpu().numpy()[:, 0]
        df_true = pd.DataFrame({'value': series_true}, index=date_range)
        result_true = seasonal_decompose(df_true['value'], model='additive')

        # Plotting
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # Create a numerical x-axis
        x_values = np.arange(len(date_range))

        # Original series
        axes[0].plot(x_values, df_pred['value'], label='Predicted', color='blue')
        axes[0].plot(x_values, df_true['value'], label='True', color='red', linestyle='--')
        axes[0].set_ylabel('Original Series', rotation=0, labelpad=80, va='center')
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Trend
        axes[1].plot(x_values, result_pred.trend, label='Predicted', color='blue')
        axes[1].plot(x_values, result_true.trend, label='True', color='red', linestyle='--')
        axes[1].set_ylabel('Trend', rotation=0, labelpad=80, va='center')
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Seasonal
        axes[2].plot(x_values, result_pred.seasonal, label='Predicted', color='blue')
        axes[2].plot(x_values, result_true.seasonal, label='True', color='red', linestyle='--')
        axes[2].set_ylabel('Seasonal', rotation=0, labelpad=80, va='center')
        axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Residual
        axes[3].plot(x_values, result_pred.resid, label='Predicted', color='blue')
        axes[3].plot(x_values, result_true.resid, label='True', color='red', linestyle='--')
        axes[3].set_ylabel('Residual', rotation=0, labelpad=80, va='center')
        axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Remove x-tick labels
        for ax in axes:
            ax.set_xticklabels([])

        fig.tight_layout()

        # Log the figure
        self.logger.experiment.log({f'{mode}/Decomposition': fig})
        plt.close(fig)

    @staticmethod
    def loss_fn(pred, trues):
        return torch.mean(torch.abs(pred - trues) ** 2)

    def training_step(self, batch, batch_idx):
        x, class_x, feat, _ = batch

        out = ForwardResult(*self.forward(x=x, feat=feat, teacher_forcing=True, results_phase=False))
        loss = self.loss_fn(out.pred_spec, out.true_spec)

        self.train_step_outputs.loss.append(loss)
        self.train_step_outputs.ForwardResult.append(out)
        
        self.train_example_pred, self.train_example_true = out.pred_time, out.trues

        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.tensor(self.train_step_outputs.loss).mean()

        self.plot_signals(
            self.train_step_outputs.ForwardResult[0].pred_time,
            self.train_step_outputs.ForwardResult[0].trues,
            teacher_forcing=self.train_step_outputs.ForwardResult[0].teacher_forcing,
            name="Train",
        )
        self.logResults(
            self.train_step_outputs.ForwardResult[0].decoder_output,
            self.train_step_outputs.ForwardResult[0].pred_spec,
            self.train_step_outputs.ForwardResult[0].trues.permute(1, 0),
            self.train_step_outputs.ForwardResult[0].filtered,
            self.train_step_outputs.ForwardResult[0].unfiltered,
            self.train_step_outputs.ForwardResult[0].teacher_forcing,
            self.train_step_outputs.ForwardResult[0].freq_importance_weights,
            self.train_step_outputs.ForwardResult[0].lambdas,
            title="Train",
        )
        self.log("Train/Loss", avg_train_loss)
        self.train_step_outputs.loss.clear()
        self.train_step_outputs.ForwardResult.clear()
        
    def validation_step(self, batch, batch_idx):
        x, class_x, feat, _ = batch

        out = ForwardResult(*self.forward(x=x, feat=feat, teacher_forcing=True, results_phase=False))
        loss = self.loss_fn(out.pred_spec, out.true_spec)

        self.validation_step_outputs.loss.append(loss)
        self.validation_step_outputs.ForwardResult.append(out)

        self.example_pred, self.example_true = out.pred_time, out.trues
        
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.tensor(self.validation_step_outputs.loss).mean()

        self.plot_signals(
            self.validation_step_outputs.ForwardResult[0].pred_time,
            self.validation_step_outputs.ForwardResult[0].trues,
            teacher_forcing=self.validation_step_outputs.ForwardResult[0].teacher_forcing,
            name="Validation",
        )
        self.logResults(
            self.validation_step_outputs.ForwardResult[0].decoder_output,
            self.validation_step_outputs.ForwardResult[0].pred_spec,
            self.validation_step_outputs.ForwardResult[0].trues.permute(1, 0),
            self.validation_step_outputs.ForwardResult[0].filtered,
            self.validation_step_outputs.ForwardResult[0].unfiltered,
            self.validation_step_outputs.ForwardResult[0].teacher_forcing,
            self.validation_step_outputs.ForwardResult[0].freq_importance_weights,
            self.validation_step_outputs.ForwardResult[0].lambdas,
            title="Validation",
        )
        self.log("Validation/Loss", avg_val_loss)
        self.validation_step_outputs.loss.clear()
        self.validation_step_outputs .ForwardResult.clear()

    def test_step(self, batch, batch_idx):
        x, class_x, feat, _ = batch

        out = ForwardResult(*self.forward(x=x, feat=feat, teacher_forcing=False, results_phase=False))
        loss = self.loss_fn(out.pred_spec, out.true_spec)

        self.test_step_outputs.loss.append(loss)
        self.test_step_outputs.ForwardResult.append(out)
        
        self.test_example_pred, self.test_example_true = out.pred_time, out.trues

        return loss

    def on_test_epoch_end(self):
        avg_test_loss = torch.tensor(self.test_step_outputs.loss).mean()
        for i in range(min(10, self.args.batch_size)):
            self.plot_signals(
                self.test_step_outputs.ForwardResult[0].pred_time[:, i:],
                self.test_step_outputs.ForwardResult[0].trues[:, i:],
                teacher_forcing=self.test_step_outputs.ForwardResult[0].teacher_forcing,
                name="Test",
            )
            self.logResults(
                self.test_step_outputs.ForwardResult[0].decoder_output[:, i:],
                self.test_step_outputs.ForwardResult[0].pred_spec[i:],
                self.test_step_outputs.ForwardResult[0].trues.permute(1, 0)[i:],
                self.test_step_outputs.ForwardResult[0].filtered[:, i:],
                self.test_step_outputs.ForwardResult[0].unfiltered[:, i:],
                self.test_step_outputs.ForwardResult[0].teacher_forcing,
                self.test_step_outputs.ForwardResult[0].freq_importance_weights[i:],
                self.test_step_outputs.ForwardResult[0].lambdas[:, i:],
                title="Test",
            )
            self.log_decomposition('Test', self.test_step_outputs.ForwardResult[0].pred_time[:, i:], self.test_step_outputs.ForwardResult[0].trues[:, i:])

        self.log("Test/Loss", avg_test_loss)
        self.test_step_outputs.loss.clear()
        self.test_step_outputs.ForwardResult.clear()

    def on_train_end(self):
        self.log_decomposition('Train', self.train_example_pred, self.train_example_true)
    
    def on_test_end(self):
        self.log_decomposition('Test', self.test_example_pred, self.test_example_true)
        
    def on_validation_end(self):
        self.log_decomposition('Validation', self.example_pred, self.example_true)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = {
            "scheduler": optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def early_stopping_criteria(self, min_delta=0.001, patience=5):
        if len(self.validation_losses) >= patience:
            recent_losses = self.validation_losses[-patience:]
            best_loss = min(recent_losses)
            if all(best_loss - loss < min_delta for loss in recent_losses):
                return True
        return False

