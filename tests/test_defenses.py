import pytest
import torch
from defenses.dynanoise import DynaNoise
from defenses.noise_transition import NoiseTransition
from defenses.adani import AdaNI
from defenses.adamixup import AdaMixup
from defenses.hyga_ani_defense import HyGAANI

@pytest.fixture
def sample_model():
    # Create a dummy model for testing
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 30 * 30, 10)
    )
    return model

@pytest.fixture
def sample_data():
    # Create a sample input tensor for testing
    return torch.randn(32, 3, 32, 32)  # 32 images, 3 channels, 32x32 resolution

def test_dynanoise_defense(sample_model, sample_data):
    defense = DynaNoise(model=sample_model)
    output = defense.apply_defense(sample_data)
    assert output.shape == (32, 10), f"Expected output shape (32, 10), but got {output.shape}"

def test_noise_transition_defense(sample_model, sample_data):
    defense = NoiseTransition(model=sample_model)
    output = defense.apply_defense(sample_data)
    assert output.shape == (32, 10), f"Expected output shape (32, 10), but got {output.shape}"

def test_adani_defense(sample_model, sample_data):
    defense = AdaNI(model=sample_model)
    output = defense.apply_defense(sample_data)
    assert output.shape == (32, 10), f"Expected output shape (32, 10), but got {output.shape}"

def test_adamixup_defense(sample_model, sample_data):
    defense = AdaMixup(model=sample_model)
    output = defense.apply_defense(sample_data)
    assert output.shape == (32, 10), f"Expected output shape (32, 10), but got {output.shape}"

def test_hygaani_defense(sample_model, sample_data):
    defense = HyGAANI(model=sample_model)
    output = defense.apply_defense(sample_data)
    assert output.shape == (32, 10), f"Expected output shape (32, 10), but got {output.shape}"
