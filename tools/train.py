from argparse import ArgumentParser

from torchsweetie.trainer import ClsTrainer


def main(cfg) -> None:
    trainer = ClsTrainer(cfg.cfg_file, cfg.run_dir)

    trainer.train()

    print("\n==================Train Finished, Starting Test==================\n")

    if cfg.best:
        prefix = "best"
    elif cfg.last:
        prefix = "last"
    else:
        prefix = "epoch"

    trainer.test(prefix, cfg.digits, cfg.export)


if __name__ == "__main__":
    parser = ArgumentParser()

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

    group_test_weights = parser.add_mutually_exclusive_group()
    group_test_weights.add_argument(
        "--best",
        action="store_true",
        help="whether to load the best weights when test",
    )
    group_test_weights.add_argument(
        "--last",
        action="store_true",
        help="whether to load the last weights when test",
    )
    group_test_weights.add_argument(
        "--epoch", type=int, help="which epoch of weights want to load when test"
    )

    parser.add_argument(
        "--digits",
        default=3,
        type=int,
        help="digits remain for accuracy when test",
    )
    parser.add_argument(
        "--export", action="store_true", help="whether to export the report after test"
    )

    main(cfg=parser.parse_args())
