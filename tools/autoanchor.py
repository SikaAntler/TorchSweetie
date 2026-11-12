from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans

from torchsweetie.data import Annotation


def main(cfg) -> None:
    train_set_path = Path(cfg.train_set)
    train_set = pd.read_csv(train_set_path, header=None)

    wh = []

    for ann_file in train_set[1]:
        annotation = Annotation.from_json(ann_file)
        for bbox in annotation.bboxes:
            wh.append((bbox.width, bbox.height))

    wh = np.array(wh)
    anchors, distortion = kmeans(wh, k_or_guess=9)
    print(f"distortion={distortion.item()}")
    anchors = np.sort(anchors, axis=0).round().astype(int)

    # for yaml format
    anchors = anchors.reshape(3, 6)
    print("anchors: &anchors")
    for i in range(3):
        w0, h0, w1, h1, w2, h2 = anchors[i]
        print(f"  - [{w0},{h0}, {w1},{h1}, {w2},{h2}]")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("train_set", type=str)

    main(parser.parse_args())
