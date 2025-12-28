import math
from argparse import ArgumentParser
from pathlib import Path

from rich import print

from torchsweetie.utils import load_config


def main(cfg) -> None:
    config = load_config(Path(cfg.cfg_file))

    batch_size = config.train_dataloader.batch_size
    print(f"{batch_size = }")

    accumulation = config.train.gradient_accumulation_steps
    print(f"{accumulation = }")

    num_batches = math.floor(cfg.num_samples / batch_size / cfg.num_processes)
    print(f"{num_batches = }")

    num_iters = num_batches * cfg.num_epochs
    print(f"{num_iters = }")

    optimize_steps = math.ceil(num_batches / accumulation) * cfg.num_epochs
    print(f"{optimize_steps = }")

    warmup_iters = math.ceil(num_batches / accumulation) * cfg.warmup
    print(f"{warmup_iters = }")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--cfg-file", "--cfg", type=str, help="path of config file")

    parser.add_argument("--num-processes", "-p", default=1, type=int, help="num of processes")

    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        required=True,
        help="the number of samples in train dataset",
    )

    parser.add_argument(
        "--num-epochs", "-e", type=int, required=True, help="the number of expected epochs"
    )

    parser.add_argument("--warmup", "-w", default=0, type=int, help="the number of warmup epochs")

    main(parser.parse_args())
