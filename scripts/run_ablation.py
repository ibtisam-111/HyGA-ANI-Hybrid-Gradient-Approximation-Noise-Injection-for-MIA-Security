import argparse
import torch
import yaml
from models.hyga_ani import HyGAANI
from attacks.mrm import MrM
from attacks.fes_mia import FeSMIA
from attacks.smile import SMILE
from attacks.imia import IMIA
from attacks.seqmia import SeqMIA
from datasets import get_dataset
from evaluation.metrics import compute_attack_metrics

def run_attack(args):
    # Load configuration
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Prepare dataset
    _, test_dataset = get_dataset(config['dataset'], train=False, val=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = HyGAANI(config['model_params'])
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Define the attack
    if args.attack_type == 'MRM':
        attack = MrM(model)
    elif args.attack_type == 'FeS-MIA':
        attack = FeSMIA(model)
    elif args.attack_type == 'SMILE':
        attack = SMILE(model)
    elif args.attack_type == 'IMIA':
        attack = IMIA(model)
    elif args.attack_type == 'SeqMIA':
        attack = SeqMIA(model)

    # Run the attack and compute metrics
    success_rate = attack.execute(test_loader)
    print(f"Attack Success Rate: {success_rate}%")

    # Optionally, save the attack results
    compute_attack_metrics(success_rate, args.attack_type, args.results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyGA-ANI Attack Execution Script")
    parser.add_argument('--config_file', type=str, default='configs/cifar10.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--attack_type', type=str, required=True, choices=['MRM', 'FeS-MIA', 'SMILE', 'IMIA', 'SeqMIA'], help='Type of attack to execute')
    parser.add_argument('--results_path', type=str, default='results/cifar10/attack_results.txt', help='Path to save attack results')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_attack(args)
