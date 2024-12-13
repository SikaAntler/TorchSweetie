from argparse import ArgumentParser

from torchsweetie.tester import ClsTester
from torchsweetie.trainer import ClsTrainer


def main(cfg) -> None:
    trainer = ClsTrainer(cfg.cfg_file, cfg.run_dir)
    trainer.train()

    if not trainer.accelerator.is_main_process:
        return

    print("\n==================Train Finished==================\n")

    cfg_file = str(trainer.cfg_file.relative_to(trainer.root_dir))
    run_dir = trainer.exp_dir.parent.parent.name
    exp_dir = trainer.exp_dir.name

    if cfg.best:
        prefix = "best"
    elif cfg.last:
        prefix = "last"
    else:
        prefix = "epoch"
    weights = list(trainer.exp_dir.glob(f"{prefix}-*[0-9].pth"))
    assert len(weights) == 1
    weights = weights[0].name

    if cfg.test:
        print(f"\n==================Starting Test==================\n")

        tester = ClsTester(cfg_file, run_dir, exp_dir, weights)
        tester.test()
        tester.report(cfg.digits, cfg.export)


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

    parser.add_argument("--test", action="store_true", help="whether to test")
    parser.add_argument(
        "--digits",
        default=3,
        type=int,
        help="digits remain for accuracy when print report after test",
    )
    parser.add_argument(
        "--export", action="store_true", help="whether to export the report after test"
    )

    main(cfg=parser.parse_args())
