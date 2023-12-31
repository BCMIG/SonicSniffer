import argparse


def get_config():
    parser = argparse.ArgumentParser()

    # arguments for main.py
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    parser.add_argument("--model_type", type=str, default="tiny")

    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    # arguments for LightningModule
    # parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # pos_weight is estimated to be ~53 by SonicSnifferDataset
    parser.add_argument("--pos_weight", type=float, default=1.0)

    # arguments for Dataset
    parser.add_argument("--num_workers", type=int, default=64)

    args = parser.parse_args()

    # https://arxiv.org/pdf/2012.12877.pdf
    # *4 because DDP with 4 GPUs
    args.lr = 0.0001 * args.batch_size * 4  #  / 512

    return args
