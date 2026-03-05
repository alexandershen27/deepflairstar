import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data import DeepFLAIRDataModule
from src.lightning_module import DeepFLAIRLightningModule

def main(args):
    # Set precision
    torch.set_float32_matmul_precision('high')
    
    # 1. DataModule
    dm = DeepFLAIRDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        num_workers=args.num_workers,
        cache_rate=args.cache_rate
    )
    
    # 2. LightningModule
    model = DeepFLAIRLightningModule(
        base_channels=args.base_channels,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        mse_weight=args.mse_weight,
        ssim_weight=args.ssim_weight,
        grad_weight=args.grad_weight,
        patch_size=(args.patch_size, args.patch_size, args.patch_size)
    )
    
    # 3. Loggers
    os.makedirs("logs", exist_ok=True)
    tb_logger = TensorBoardLogger("logs", name="deepflair_tb")
    ml_logger = MLFlowLogger(experiment_name="DeepFLAIR_Star", save_dir="logs/mlflow")
    
    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="outputs/checkpoints",
        filename="deepflair-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        logger=[tb_logger, ml_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        precision="16-mixed" if args.devices != "cpu" else 32,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else None,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else None,
    )
    
    # 6. Train
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=75)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 full batch for train/val/test to catch bugs")
    parser.add_argument("--limit_train_batches", type=int, default=0, help="Limit number of train batches for testing")
    parser.add_argument("--limit_val_batches", type=int, default=0, help="Limit number of val batches for testing")
    
    args = parser.parse_args()
    main(args)
