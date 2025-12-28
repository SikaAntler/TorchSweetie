from abc import ABC, abstractmethod
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from ..utils import TRANSFORMS, BoxFormat
from .det_datastructs import Annotation, DetDataImage, DetDataPack, DetDataTensor


class DetTransform(nn.Module, ABC):
    dataset: list[tuple[str, str]]

    @abstractmethod
    def __call__(self, data: DetDataImage) -> DetDataImage: ...


class DetDataset(Dataset):
    SCOPE = "detection"

    def __init__(
        self,
        dataset_file: str,
        box_format: BoxFormat,
        classes_file: str | None = None,
        continue_if_label_mismatch: bool = True,
        transforms: list[DictConfig] | None = None,
    ) -> None:
        super().__init__()

        self.dataset: list[tuple[str, str]] = []
        dataset = pd.read_csv(dataset_file, header=None)
        for img_file, ann_file in dataset.itertuples(False, None):
            self.dataset.append((img_file, ann_file))

        self.box_format = box_format

        if classes_file is None:
            self.classes = None
        else:
            self.classes = pd.read_csv(classes_file, header=None)[0].to_list()

        self.continue_if_label_mismatch = continue_if_label_mismatch

        self.transforms: list[DetTransform] = []
        if transforms is not None:
            for cfg in transforms:
                if "scope" not in cfg:
                    cfg.scope = self.SCOPE
                transform = TRANSFORMS.create(cfg)
                assert isinstance(transform, DetTransform)
                transform.dataset = self.dataset
                self.transforms.append(transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DetDataTensor:  # ty: ignore
        img_file, ann_file = self.dataset[idx]

        image = self.load_image(img_file)
        H, W = image.shape[:2]

        annotation = Annotation.from_json(ann_file)

        data = DetDataImage(image, (W, H), deepcopy(annotation.bboxes))

        for t in self.transforms:
            data = t(data)

        image = data.image.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        image = image.astype(np.float32)
        image /= 255

        img_h, img_w = image.shape[-2:]

        cls_idxs: list[int] = []
        boxes: list[tuple[float, float, float, float]] = []
        for bbox in data.bboxes:
            if (
                self.classes is not None
                and bbox.label not in self.classes
                and self.continue_if_label_mismatch
            ):
                continue
            # 浮点边界坐标：横向统一除以 W，纵向统一除以 H。
            cls_idxs.append(0 if self.classes is None else self.classes.index(bbox.label))
            match self.box_format:
                case BoxFormat.xyxy:
                    boxes.append(
                        (
                            bbox.left / img_w,
                            bbox.top / img_h,
                            bbox.right / img_w,
                            bbox.bottom / img_h,
                        )
                    )
                case BoxFormat.xywh:
                    boxes.append(
                        (
                            bbox.left / img_w,
                            bbox.top / img_h,
                            bbox.width / img_w,
                            bbox.height / img_h,
                        )
                    )
                case BoxFormat.cxcywh:
                    boxes.append(
                        (
                            bbox.center_x / img_w,
                            bbox.center_y / img_h,
                            bbox.width / img_w,
                            bbox.height / img_h,
                        )
                    )

        return DetDataTensor(
            torch.tensor(image, dtype=torch.float32),
            data.ori_size,
            torch.LongTensor(cls_idxs),
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        )

    @staticmethod
    def collate_fn(batch_list: list[DetDataTensor]) -> DetDataPack:
        img_idxs = []
        images = []
        ori_sizes = []
        cls_idxs = []
        boxes = []

        for i, data in enumerate(batch_list):
            img_idxs.extend([i] * len(data.cls_idxs))
            images.append(data.image)
            ori_sizes.append(data.ori_size)
            cls_idxs.append(data.cls_idxs)
            boxes.append(data.boxes)

        return DetDataPack(
            torch.LongTensor(img_idxs),
            torch.stack(images),
            ori_sizes,
            torch.cat(cls_idxs),
            torch.cat(boxes),
        )

    @staticmethod
    def load_image(img_file: str) -> np.ndarray:
        image = cv2.imread(img_file, cv2.IMREAD_COLOR_RGB)  # (H, W, C)
        assert isinstance(image, np.ndarray)
        H, W = image.shape[:2]
        if len(image.shape) == 2:
            image = image.reshape(H, W, 1).repeat(3, 2)

        return image
