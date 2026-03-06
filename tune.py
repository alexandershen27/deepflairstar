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
        gpu_id = trial.number % torch.cuda.device_count()
        devices = [gpu_id]
    else:
        devices = 1

    # 2. Hyperparameter Suggestions
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    ssim_weight = trial.suggest_float("ssim_weight", 1.0, 10.0, log=True)
    grad_weight = trial.suggest_float("grad_weight", 1.0, 5.0)
    
    # 3. Data & Model
    dm = DeepFLAIRDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers // args.n_jobs,
        num_samples=8,
        cache_rate=1.0
    )
    
    model = DeepFLAIRLightningModule(
        base_channels=args.base_channels,
        lr=lr,
        ssim_weight=ssim_weight,
        grad_weight=grad_weight
    )
    
    # Robust Manual Weight Loading
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('model.'): name = name.replace('model.', '', 1)
        if name.startswith('module.'): name = name.replace('module.', '', 1)
        if name.startswith('model.'): name = name.replace('model.', '', 1) # Double-check prefix
        new_state_dict[name] = v
    model.model.load_state_dict(new_state_dict, strict=False)

    # 4. Trainer (Short Trial)
    # We enable the MLflow logger so we can see the artifacts for EACH trial
    log_dir = os.path.abspath("logs/mlflow")
    ml_logger = pl.loggers.MLFlowLogger(
        experiment_name="DeepFLAIR_Tuning", 
        save_dir=log_dir,
        tags={"trial_number": str(trial.number)}
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=devices,
        enable_checkpointing=False,
        logger=ml_logger,
        precision="16-mixed",
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    
    trainer.fit(model, datamodule=dm)
    return trainer.callback_metrics["val_loss"].item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    # Create MLflow Callback
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

    print("\n--- TUNING COMPLETE ---")
    print(f"Best parameters: {study.best_params}")

if __name__ == "__main__":
    main()
