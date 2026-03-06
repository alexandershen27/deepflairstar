import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data import DeepFLAIRDataModule
from src.lightning_module import DeepFLAIRLightningModule

# 1. Callbacks for continuity and metrics
class ResumeEpochCallback(pl.Callback):
    def __init__(self, start_epoch):
        self.start_epoch = start_epoch
    def on_train_start(self, trainer, pl_module):
        print(f"--- CALLBACK: Forcing Trainer state to Epoch {self.start_epoch} ---")
        trainer.fit_loop.epoch_progress.current.completed = self.start_epoch
        trainer.fit_loop.epoch_progress.current.processed = self.start_epoch

# 2. Callback to safely start system metrics logging in DDP mode
class SystemMetricsCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            try:
                import mlflow
                ml_logger = next((l for l in trainer.loggers if isinstance(l, MLFlowLogger)), None)
                if ml_logger:
                    print(f"--- SYSTEM METRICS: Starting monitor for Run {ml_logger.run_id} ---")
                    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
                    mlflow.enable_system_metrics_logging()
            except: pass

def main(args):
    # 0. Reproducibility
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    
    # 1. DataModule
    dm = DeepFLAIRDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        num_samples=args.num_samples
    )
    
    # 2. LightningModule
    model = DeepFLAIRLightningModule(
        model_type=args.model_type, # Now dynamic
        base_channels=args.base_channels,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        mse_weight=args.mse_weight,
        ssim_weight=args.ssim_weight,
        grad_weight=args.grad_weight,
        patch_size=(args.patch_size, args.patch_size, args.patch_size)
    )
    
    start_epoch = 0
    # Manual Weight Loading
    if args.ckpt_path:
        print(f"--- MANUAL LOAD: Transferring weights from {args.ckpt_path} ---")
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        start_epoch = checkpoint.get('epoch', 0)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('model.'): name = name.replace('model.', '', 1)
            if name.startswith('module.'): name = name.replace('module.', '', 1)
            if name.startswith('model.'): name = name.replace('model.', '', 1) # Double-check prefix
            new_state_dict[name] = v
            
        model.model.load_state_dict(new_state_dict, strict=False)
    
    # 3. Loggers
    log_dir = os.path.abspath("logs/mlflow")
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{log_dir}")
        mlflow.set_experiment("DeepFLAIR_Star")
    except:
        pass

    tb_logger = TensorBoardLogger("logs", name="deepflair_tb")
    ml_logger = MLFlowLogger(
        experiment_name="DeepFLAIR_Star", 
        save_dir=log_dir,
        log_model=True
    )
    
    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="outputs/checkpoints",
        filename="deepflair-{epoch:03d}-{val_loss:.4f}",
        save_top_k=10,
        mode="min",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    callbacks = [checkpoint_callback, lr_monitor, SystemMetricsCallback()]
    if start_epoch > 0:
        callbacks.append(ResumeEpochCallback(start_epoch=start_epoch))
    
    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        strategy=args.strategy,
        sync_batchnorm=True,
        logger=[tb_logger, ml_logger],
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed" if args.devices != "cpu" else 32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    
    # 6. Train
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet", "unetr"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_val_batches", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default=None)
    
    args = parser.parse_args()
    main(args)
