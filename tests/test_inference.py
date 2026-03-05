import torch
from src.lightning_module import DeepFLAIRLightningModule

def test_sliding_window_reconstruction():
    """Verify that SlidingWindowInferer reconstructs full volume from patches."""
    patch_size = (32, 32, 32)
    # Target volume size (smaller than real brain for speed)
    volume_size = (64, 80, 64)
    
    model = DeepFLAIRLightningModule(base_channels=4, patch_size=patch_size)
    model.eval()
    
    dummy_volume = torch.randn(1, 1, *volume_size)
    
    with torch.no_grad():
        output = model.inferer(dummy_volume, model)
        
    assert output.shape == dummy_volume.shape
    # Check that it's not all zeros
    assert torch.abs(output).sum() > 0
