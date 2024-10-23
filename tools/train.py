from argparse import ArgumentParser

from torchsweetie.trainer import ClsTrainer


def main(cfg) -> None:
    trainer = ClsTrainer(cfg.root_dir, cfg.cfg_file, cfg.run_dir)

    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--root-dir",
        "--root",
        type=str,
        required=True,
        help="path of the root directory",
    )
    parser.add_argument(
        "--cfg-file",
        "--cfg",
        type=str,
        required=True,
        help="path of the config file (relative)",
    )
    parser.add_argument(
        "--run-dir",
        "--run",
        default="runs",
        type=str,
        help="path of the running directory (relative)",
    )

    main(cfg=parser.parse_args())
