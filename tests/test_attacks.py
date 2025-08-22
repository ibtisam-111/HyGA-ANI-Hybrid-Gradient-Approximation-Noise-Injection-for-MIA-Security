import pytest
from attacks.mrm import MrM
from attacks.fes_mia import FeSMIA
from attacks.smile import SMILE
from attacks.imia import IMIA
from attacks.seqmia import SeqMIA

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

def test_mrm_attack(sample_model, sample_data):
    attack = MrM(model=sample_model)
    success_rate = attack.run_attack(sample_data)
    assert success_rate > 0.5, f"Expected success rate to be > 0.5, but got {success_rate}"

def test_fesmia_attack(sample_model, sample_data):
    attack = FeSMIA(model=sample_model)
    success_rate = attack.run_attack(sample_data)
    assert success_rate > 0.5, f"Expected success rate to be > 0.5, but got {success_rate}"

def test_smile_attack(sample_model, sample_data):
    attack = SMILE(model=sample_model)
    success_rate = attack.run_attack(sample_data)
    assert success_rate > 0.5, f"Expected success rate to be > 0.5, but got {success_rate}"

def test_imia_attack(sample_model, sample_data):
    attack = IMIA(model=sample_model)
    success_rate = attack.run_attack(sample_data)
    assert success_rate > 0.5, f"Expected success rate to be > 0.5, but got {success_rate}"

def test_seqmia_attack(sample_model, sample_data):
    attack = SeqMIA(model=sample_model)
    success_rate = attack.run_attack(sample_data)
    assert success_rate > 0.5, f"Expected success rate to be > 0.5, but got {success_rate}"
