import argparse
import os
import torch
import optuna
from optuna.integration.mlflow import MLflowCallback
import pytorch_lightning as pl
from src.data import DeepFLAIRDataModule
from src.lightning_module import DeepFLAIRLightningModule

def objective(trial, args):
    # 1. GPU Isolation for Parallel Trials
    if args.n_jobs > 1:
        # We assume 4 GPUs available
        gpu_id = trial.number % torch.cuda.device_count()
        devices = [gpu_id]
    else:
        devices = 1

    # 2. Hyperparameter Suggestions
    lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
    ssim_weight = trial.suggest_float("ssim_weight", 1.0, 10.0, log=True)
    grad_weight = trial.suggest_float("grad_weight", 1.0, 5.0)
    
    # 3. Data & Model
    dm = DeepFLAIRDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers // args.n_jobs,
        num_samples=args.num_samples,
        cache_rate=1.0
    )
    
    model = DeepFLAIRLightningModule(
        base_channels=args.base_channels,
        lr=lr,
        ssim_weight=ssim_weight,
        grad_weight=grad_weight,
        patch_size=(args.patch_size, args.patch_size, args.patch_size)
    )
    
    # 4. Logging & Trainer
    log_dir = os.path.abspath("logs/mlflow")
    ml_logger = pl.loggers.MLFlowLogger(
        experiment_name="DeepFLAIR_Tuning", 
        save_dir=log_dir,
        tags={"trial_number": str(trial.number), "gpu": str(devices)}
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=devices,
        enable_checkpointing=False,
        logger=ml_logger,
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=False,
        enable_model_summary=False
    )
    
    trainer.fit(model, datamodule=dm)
    return trainer.callback_metrics["val_loss"].item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    # Create MLflow Callback for Optuna
    log_dir = os.path.abspath("logs/mlflow")
    mlflc = MLflowCallback(
        tracking_uri=f"file:{log_dir}",
        metric_name="val_loss",
    )

    study = optuna.create_study(
        direction="minimize",
        study_name="deepflair_tune",
        storage="sqlite:///optuna.db",
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective(trial, args), 
        n_trials=args.n_trials, 
        n_jobs=args.n_jobs,
        callbacks=[mlflc]
    )

    print("\n" + "="*40)
    print("--- TUNING COMPLETE ---")
    print(f"Best parameters: {study.best_params}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
