import torch
import pytest
from src.arch.unet import DeepFLAIRNet
from src.losses import DeepFLAIRLoss, GradientLoss

def test_model_forward():
    """Verify output shape matches input shape."""
    model = DeepFLAIRNet(in_channels=1, out_channels=1, base_channels=8) # Small base for test speed
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    output = model(dummy_input)
    assert output.shape == dummy_input.shape
    assert torch.all(output >= 0.0) # ReLU at end

def test_loss_zero():
    """Loss should be zero (or very close) for identical tensors."""
    loss_fn = DeepFLAIRLoss()
    x = torch.rand(1, 1, 32, 32, 32)
    total_loss, metrics = loss_fn(x, x)
    assert total_loss.item() < 1e-6
    assert metrics["mse"].item() < 1e-6
    assert metrics["ssim_loss"].item() < 1e-5 # SSIM might have small eps
    assert metrics["grad_loss"].item() < 1e-6

def test_gradient_loss():
    """Gradient loss should penalize differences in edges."""
    grad_loss = GradientLoss()
    x = torch.zeros(1, 1, 32, 32, 32)
    y = torch.zeros(1, 1, 32, 32, 32)
    
    # Same tensors = zero loss
    assert grad_loss(x, y).item() == 0.0
    
    # Introduce a single voxel difference (small impact on grad)
    y[0, 0, 16, 16, 16] = 1.0
    loss_single = grad_loss(x, y).item()
    assert loss_single > 0
    
    # Introduce a different pattern
    z = torch.rand(1, 1, 32, 32, 32)
    loss_rand = grad_loss(x, z).item()
    assert loss_rand > loss_single

def test_model_parameters():
    """Check if model complexity scales with base_channels."""
    model_8 = DeepFLAIRNet(base_channels=8)
    num_params_8 = sum(p.numel() for p in model_8.parameters())
    
    model_4 = DeepFLAIRNet(base_channels=4)
    num_params_4 = sum(p.numel() for p in model_4.parameters())
    
    assert num_params_8 > num_params_4
    assert num_params_4 > 0
