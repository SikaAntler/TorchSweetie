from enum import StrEnum


class BoxFormat(StrEnum):
    xyxy = "xyxy"
    xywh = "xywh"
    cxcywh = "cxcywh"
