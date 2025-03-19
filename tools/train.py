from argparse import ArgumentParser

from torchsweetie.exporter import RetrievalExporter
from torchsweetie.tester import ClsTester, RetrievalTester
from torchsweetie.trainer import ClsTrainer


def main(cfg) -> None:
    trainer = ClsTrainer(cfg.cfg_file, cfg.run_dir)
    trainer.train()

    if not trainer.accelerator.is_main_process:
        return

    print("\n==================Train Finished==================\n")

    if cfg.best:
        weights = "best-*[0-9].pth"
    elif cfg.last:
        weights = "last-*[0-9].pth"
    else:
        weights = f"epoch-{cfg.epoch}.pth"

    weights = list(trainer.exp_dir.glob(weights))
    assert len(weights) == 1
    weights = weights[0].name

    exp_dir = str(trainer.exp_dir.relative_to(trainer.root_dir))
    if cfg.test:
        print(f"\n==================Starting Test==================\n")

        tester = ClsTester(cfg.cfg_file, exp_dir, weights)
        tester.test()
        tester.report(cfg.digits, cfg.export)
    elif cfg.retrieval:
        print(f"\n==================Starting Retrieval==================\n")

        exporter = RetrievalExporter(cfg.cfg_file, exp_dir, weights)
        exporter.export()
        tester = RetrievalTester(cfg.cfg_file, exp_dir, weights)
        tester.test()
        tester.report(exporter.embeddings, exporter.labels, cfg.topk_list, cfg.digits)


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

    group_test_type = parser.add_mutually_exclusive_group()
    group_test_type.add_argument("--test", action="store_true", help="whether to test")
    group_test_type.add_argument("--retrieval", action="store_true", help="whether to retrieval")

    group_test_weights = parser.add_mutually_exclusive_group()
    group_test_weights.add_argument(
        "--best",
        action="store_true",
        help="whether to load the best weights when test or retrieval",
    )
    group_test_weights.add_argument(
        "--last",
        action="store_true",
        help="whether to load the last weights when test or retrieval",
    )
    group_test_weights.add_argument(
        "--epoch", type=int, help="which epoch of weights want to load when test or retrieval"
    )

    parser.add_argument(
        "--digits",
        default=3,
        type=int,
        help="digits remain for accuracy when print report after test or retrieval",
    )
    parser.add_argument(
        "--export", action="store_true", help="whether to export the report after test"
    )
    parser.add_argument(
        "--topk-list", "--topk", nargs="+", type=int, help="the list of k in topk when retrieval"
    )

    main(cfg=parser.parse_args())
