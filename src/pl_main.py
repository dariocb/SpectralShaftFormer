import wandb
from arguments import load_args, parse_args
from pl_dataloaders import DataModule
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_TransformerModel import SSF

def get_next_folder_id(base_path, folder_prefix):
    # List all directories in the base path
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Filter directories that match the folder prefix and extract their numerical ID
    existing_ids = []
    for d in dirs:
        if d.startswith(folder_prefix):
            try:
                folder_id = int(d[len(folder_prefix):])
                existing_ids.append(folder_id)
            except ValueError:
                continue
    
    # Sort the extracted IDs
    existing_ids.sort()
    
    # Keep only the largest ID
    largest_id = max(existing_ids) if existing_ids else -1

    return largest_id + 1


if __name__ == "__main__":
    group_name="JulyRevision"
    
    args, data_args = parse_args()


    if not os.path.exists("./results/"):
        os.mkdir("./results/")

    # i = 0
    # while os.path.exists(f"./results/{args.name_folder}"):
    #     i += 1
    #     args.name_folder = f"Exp{i}"
    # os.mkdir(f"./results/{args.name_folder}")

    if args.resume_run:
        load_args(args)
    else:
        base_path = "./results"
        folder_prefix = "Exp"

        next_id = get_next_folder_id(base_path, folder_prefix)
        args.name_folder = f"{folder_prefix}{next_id}"

        # Create the new directory
        new_folder_path = os.path.join(base_path, args.name_folder)
        os.mkdir(new_folder_path)

    run_name=args.name_folder

    wandb_logger = WandbLogger(
        project="Trenecitos",
        group=group_name,
        name=run_name,
        config=args,
        log_model=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        accelerator="cpu",
        devices='auto',
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True),
            pl.callbacks.EarlyStopping(
                monitor="Validation/Loss",
                patience=15,
                mode="min",
                verbose=True,
                min_delta=0.0001,
            ),
            # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        logger=wandb_logger,
        # inference_mode=False,  # always compute gradients
    )

    model = SSF(args, data_args)
    data_module = DataModule(args, data_args)

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path="best")

    # wandb.save(f'./results/Exp{i}/checkpoint.pth')



# source /export/usuarios01/dcabezas/miniconda3/bin/activate Trenecitos
