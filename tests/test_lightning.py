import torch
import pytorch_lightning as pl
from src.lightning_module import DeepFLAIRLightningModule
from monai.data import Dataset, DataLoader

class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, patch_size=(32, 32, 32)):
        super().__init__()
        self.patch_size = patch_size

    def setup(self, stage=None):
        # Create 1 single batch of synthetic data
        # Data must have at least some variance for BatchNorm if in train mode
        # or we just use a larger patch. 32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3 (Bottleneck)
        image = torch.rand(1, 1, *self.patch_size)
        label = image * 0.5 + 0.1
        
        self.data = [{"image": image[0], "label": label[0]}]
        self.ds = Dataset(data=self.data)

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=1)

def test_overfit_synthetic():
    """Verify the model can overfit on synthetic data in seconds."""
    patch_size = (32, 32, 32) 
    dm = SyntheticDataModule(patch_size=patch_size)
    
    # Use extremely small model for test
    model = DeepFLAIRLightningModule(
        base_channels=4, 
        patch_size=patch_size,
        lr=1e-3,
        mse_weight=1.0,
        ssim_weight=0.0,
        grad_weight=0.0
    )
    
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0
    )
    
    trainer.fit(model, datamodule=dm)
    
    final_loss = trainer.callback_metrics["train_loss_epoch"].item()
    print(f"Synthetic test loss: {final_loss}")
    assert final_loss < 0.1
