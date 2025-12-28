import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans

from torchsweetie.data import Annotation


def main(cfg) -> None:
    directory = Path(cfg.directory)
    ann_files_list = list(directory.iterdir())

    wh = []

    for ann_file in ann_files_list:
        annotation = Annotation.from_json(ann_file)
        for bbox in annotation.bboxes:
            wh.append((bbox.width, bbox.height))

    wh = np.array(wh)
    anchors, distortion = kmeans(wh, k_or_guess=9)
    print(f"{distortion = }")
    anchors = np.sort(anchors, axis=0).round().astype(int)
    print(anchors.dtype)

    # for yaml format
    anchors = anchors.reshape(3, 6)
    print(f"anchors: &anchors")
    for i in range(3):
        w0, h0, w1, h1, w2, h2 = anchors[i]
        print(f"  - [{w0},{h0}, {w1},{h1}, {w2},{h2}]")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("directory", type=str)

    main(parser.parse_args())
