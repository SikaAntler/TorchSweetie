import random
from pathlib import Path
from typing import Union

import pandas as pd
from rich import print

from .smart_sort import smart_sort


def split_dataset(
    images_dir: Union[Path, str],
    target_names: list,
    train_val: tuple[int, int],
    seed: int,
    save_all_set: bool = False,
    train_set_stem: str = "train_set",
    val_set_stem: str = "val_set",
    all_set_stem: str = "all_set",
):
    if isinstance(images_dir, str):
        images_dir = Path(images_dir)
    directory = images_dir.parent
    train, val = train_val
    train_set, val_set, all_set = [], [], []

    for name in target_names:
        img_dir = images_dir / name
        label = target_names.index(name)

        img_list = list(img_dir.iterdir())
        img_list = smart_sort(img_list)

        num_train = int(len(img_list) * train / (train + val))
        random.seed(seed)
        random.shuffle(img_list)

        for filename in img_list[:num_train]:
            train_set.append((filename, label))
        for filename in img_list[num_train:]:
            val_set.append((filename, label))
        if save_all_set:
            for filename in img_list:
                all_set.append((filename, label))

    print(f"Train set: {len(train_set)}")
    print(f"  Val set: {len(val_set)}")
    if save_all_set:
        print(f"  All set: {len(all_set)}")

    train_set = pd.DataFrame(train_set)
    train_set.to_csv(directory / f"{train_set_stem}.csv", header=False, index=False)
    val_set = pd.DataFrame(val_set)
    val_set.to_csv(directory / f"{val_set_stem}.csv", header=False, index=False)
    if save_all_set:
        all_set = pd.DataFrame(all_set)
        all_set.to_csv(directory / f"{all_set_stem}.csv", header=False, index=False)

    print(f"Finished")
