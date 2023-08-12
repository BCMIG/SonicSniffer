import argparse


def get_config():
    parser = argparse.ArgumentParser()

    # arguments for main.py
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    parser.add_arguement("--model_type", type=str, default="b0")

    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    # arguments for LightningModule
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # pos_weight is estimated to be ~53 by SonicSnifferDataset
    parser.add_argument("--pos_weight", type=float, default=1.0)

    # arguments for Dataset
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    return args
