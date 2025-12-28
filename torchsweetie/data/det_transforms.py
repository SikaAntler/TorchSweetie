import math
import random
from dataclasses import replace
from typing import Sequence, override

import cv2
import numpy as np

from ..utils import TRANSFORMS
from .det_dataset import DetDataset, DetTransform
from .det_datastructs import Annotation, BBox, DetDataImage

SCOPE = "detection"


@TRANSFORMS.register(scope=SCOPE)
class Mosaic(DetTransform):
    def __init__(
        self,
        img_size: int,
        wh_thresh: float = 16,
        lwr_thresh: float = 100,
        area_thresh: float = 0.1,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.mosaic_border = [-img_size // 2, -img_size // 2]

        self.wh_thresh = wh_thresh
        self.lwr_thresh = lwr_thresh
        self.area_thresh = area_thresh

    @override
    def __call__(self, data: DetDataImage) -> DetDataImage:
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)

        indices = [-1] + random.sample(range(len(self.dataset)), 3)
        random.shuffle(indices)

        bboxes4: list[BBox] = []

        for i, index in enumerate(indices):
            if index == -1:
                img = data.image
                bboxes = data.bboxes
            else:
                img_file, ann_file = self.dataset[index]
                img = DetDataset.load_image(img_file)
                bboxes = Annotation.from_json(ann_file).bboxes

            h, w, c = img.shape

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, c), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = (max(xc - w, 0), max(yc - h, 0), xc, yc)
                x1b, y1b, x2b, y2b = (w - (x2a - x1a), h - (y2a - y1a), w, h)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            for bbox in bboxes:
                w1, h1 = bbox.width, bbox.height
                area1 = w1 * h1

                left = min(s * 2, max(0, bbox.left + padw))
                top = min(s * 2, max(0, bbox.top + padh))
                right = min(s * 2, max(0, bbox.right + padw))
                bottom = min(s * 2, max(0, bbox.bottom + padh))
                w2 = right - left + 1
                h2 = bottom - top + 1
                area2 = w2 * h2

                lwr = max(w2 / (h2 + 1e-6), h2 / (w2 + 1e-6))

                if (
                    w2 < self.wh_thresh
                    or h2 < self.wh_thresh
                    or area2 / (area1 + 1e-6) < self.area_thresh
                    or lwr > self.lwr_thresh
                ):
                    continue

                bbox = replace(bbox, left=left, top=top, right=right, bottom=bottom)
                bboxes4.append(bbox)

        data.image = img4
        data.bboxes = bboxes4

        return data


@TRANSFORMS.register(scope=SCOPE)
class RandomCrop(DetTransform):
    def __init__(self, target_size: int | Sequence[int], skip_empty: bool = True) -> None:
        super().__init__()

        if isinstance(target_size, int):
            self.target_w = target_size
            self.target_h = target_size
        else:
            assert len(target_size) == 2
            self.target_w = target_size[0]
            self.target_h = target_size[1]

        self.skip_empty = skip_empty

    @override
    def __call__(self, data: DetDataImage) -> DetDataImage:
        img_h, img_w = data.image.shape[:2]
        assert img_w >= self.target_w and img_h >= self.target_h

        data.ori_size = (self.target_w, self.target_h)

        while True:
            new_x = random.randint(0, img_w - self.target_w)
            new_y = random.randint(0, img_h - self.target_h)

            bboxes: list[BBox] = []
            for bbox in data.bboxes:
                if (
                    new_x <= bbox.center_x <= new_x + self.target_w - 1
                    and new_y <= bbox.center_y <= new_y + self.target_h - 1
                ):
                    bboxes.append(
                        replace(
                            bbox,
                            left=bbox.left - new_x,
                            top=bbox.top - new_y,
                            right=bbox.right - new_x,
                            bottom=bbox.bottom - new_y,
                        )
                    )

            if len(data.bboxes) == 0 or len(bboxes) != 0 or not self.skip_empty:
                data.image = data.image[
                    new_y : new_y + self.target_h, new_x : new_x + self.target_w
                ]
                assert data.image.shape[:2] == data.ori_size
                data.bboxes = bboxes
                break

        return data


@TRANSFORMS.register(scope=SCOPE)
class RandomHSV(DetTransform):
    def __init__(self, hgain: float, sgain: float, vgain: float) -> None:
        super().__init__()

        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    @override
    def __call__(self, data: DetDataImage) -> DetDataImage:
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(data.image, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        data.image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)

        return data


@TRANSFORMS.register(scope=SCOPE)
class RandomHorizontalFlip(DetTransform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()

        self.p = p

    @override
    def __call__(self, data: DetDataImage) -> DetDataImage:
        if random.random() >= self.p:
            data.image = cv2.flip(data.image, 1)
            W = data.image.shape[1]
            for bbox in data.bboxes:
                new_left = W - 1 - bbox.right
                new_right = W - 1 - bbox.left
                bbox.left = new_left
                bbox.right = new_right

        return data


@TRANSFORMS.register(scope=SCOPE)
class RandomPerspective(DetTransform):
    def __init__(
        self,
        img_size: int | list[int],
        degrees: float = 0,
        translate: float = 0,
        scale: float = 0,
        shear: float = 0,
        perspective: float = 0,
        wh_thresh: float = 16,
        lwr_thresh: float = 100,
        area_thresh: float = 0.1,
    ) -> None:
        super().__init__()

        self.img_size: tuple[int, int]
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = (img_size[0], img_size[1])

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

        self.wh_thresh = wh_thresh
        self.lwr_thresh = lwr_thresh
        self.area_thresh = area_thresh

    @override
    def __call__(self, data: DetDataImage) -> DetDataImage:
        W, H = self.img_size

        # Center
        C = np.eye(3)
        C[0, 2] = -data.image.shape[1] / 2
        C[1, 2] = -data.image.shape[0] / 2

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D((0, 0), a, s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * W
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * H

        M = T @ S @ R @ P @ C

        if (M != np.eye(3)).any():
            if self.perspective:
                data.image = cv2.warpPerspective(
                    data.image, M, self.img_size, borderValue=(114, 114, 114)
                )
            else:
                data.image = cv2.warpAffine(
                    data.image, M[:2], self.img_size, borderValue=(114, 114, 114)
                )

        if n := len(data.bboxes):
            bboxes = []

            targets = np.array(
                [(bbox.left, bbox.top, bbox.right, bbox.bottom) for bbox in data.bboxes]
            )
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3]) if self.perspective else xy[:, :2].reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            new[:, [0, 2]] = new[:, [0, 2]].clip(9, W)
            new[:, [1, 3]] = new[:, [1, 3]].clip(9, H)

            for i in range(n):
                bbox = data.bboxes[i]
                w1 = bbox.width * s
                h1 = bbox.height * s
                area1 = w1 * h1

                left = min(W, max(0, new[i, 0]))
                top = min(H, max(0, new[i, 1]))
                right = min(W, max(0, new[i, 2]))
                bottom = min(H, max(0, new[i, 3]))
                w2 = right - left + 1
                h2 = bottom - top + 1
                area2 = w2 * h2

                lwr = np.maximum(w2 / (h2 + 1e-6), h2 / (w2 + 1e-6))

                if (
                    w2 < self.wh_thresh
                    or h2 < self.wh_thresh
                    or area2 / (area1 + 1e-6) < self.area_thresh
                    or lwr > self.lwr_thresh
                ):
                    continue

                bbox = replace(bbox, left=left, top=top, right=right, bottom=bottom)
                bboxes.append(bbox)

            data.bboxes = bboxes

        return data


@TRANSFORMS.register(scope=SCOPE)
class RandomVerticalFlip(DetTransform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()

        self.p = p

    @override
    def __call__(self, data: DetDataImage) -> DetDataImage:
        if random.random() >= self.p:
            data.image = cv2.flip(data.image, 0)
            H = data.image.shape[0]
            for bbox in data.bboxes:
                new_top = H - 1 - bbox.bottom
                new_botom = H - 1 - bbox.top
                bbox.top = new_top
                bbox.bottom = new_botom

        return data


@TRANSFORMS.register(scope=SCOPE)
class Resize(DetTransform):
    def __init__(self, target_size: int | Sequence[int]) -> None:
        super().__init__()

        if isinstance(target_size, int):
            self.target_w = target_size
            self.target_h = target_size
        else:
            assert len(target_size) == 2
            self.target_w = target_size[0]
            self.target_h = target_size[1]

    @override
    def __call__(self, data: DetDataImage) -> DetDataImage:
        img_h, img_w, _ = data.image.shape
        data.image = cv2.resize(data.image, (self.target_w, self.target_h))

        w_ratio = self.target_w / img_w
        h_ratio = self.target_h / img_h

        for bbox in data.bboxes:
            bbox.left *= w_ratio
            bbox.top *= h_ratio
            bbox.right *= w_ratio
            bbox.bottom *= h_ratio

        return data
