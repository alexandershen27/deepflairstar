import argparse
import os
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data import DeepFLAIRDataModule
from src.lightning_module import DeepFLAIRLightningModule

def main(args):
    # Fixed seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Performance boost for modern GPUs
    torch.set_float32_matmul_precision('high')
    
    patch_size_tuple = (int(args.patch_size), int(args.patch_size), int(args.patch_size))
    
    dm = DeepFLAIRDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=patch_size_tuple,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        num_samples=args.num_samples,
        sampling_type=args.sampling_type
    )
    
    model = DeepFLAIRLightningModule(
        model_type=args.model_type,
        base_channels=args.base_channels,
        lr=args.lr,
        mse_weight=args.mse_weight,
        ssim_weight=args.ssim_weight,
        grad_weight=args.grad_weight,
        patch_size=patch_size_tuple
    )
    
    log_dir = os.path.abspath("logs/mlflow")
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
    
    exp_name = args.experiment_name or f"DF_{args.model_type}_{args.sampling_type}_BS{args.batch_size}"
    
    tb_logger = TensorBoardLogger("logs", name=exp_name)
    ml_logger = MLFlowLogger(experiment_name=exp_name, save_dir=log_dir, log_model=True)
    
    checkpoint_dir = os.path.join("outputs", exp_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="deepflair-{epoch:03d}-{val_loss:.4f}",
        save_top_k=5,
        mode="min",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = [int(devices)]
    elif isinstance(devices, str) and "," in devices:
        devices = [int(d.strip()) for d in devices.split(",")]
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        sync_batchnorm=True,
        logger=[tb_logger, ml_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    
    if args.test:
        if args.ckpt_path is None:
            raise ValueError("You must provide --ckpt_path when running with --test")
        trainer.test(model, datamodule=dm, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    # Force 'spawn' method for DDP stability on Python 3.14/Ubuntu
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser()
    # Data params
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sampling_type", type=str, default="grid", choices=["random", "grid"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4) # Balanced default
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--cache_rate", type=float, default=0.0)
    
    # Model params
    parser.add_argument("--model_type", type=str, default="unet")
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Loss weights
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=1.0)
    
    # Training runtime
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    main(args)
