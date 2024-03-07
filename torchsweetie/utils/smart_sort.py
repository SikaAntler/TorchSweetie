from pathlib import Path

from natsort import natsort
from pypinyin import Style, pinyin


def smart_sort(items: list) -> list:
    items_with_pinyin = []
    for item in items:
        py_list = pinyin(str(item), Style.NORMAL)
        py_concat = "".join(py[0] for py in py_list)
        items_with_pinyin.append((item, py_concat))
    sorted_items = natsort.natsorted(items_with_pinyin, key=lambda x: x[-1])

    return [item[0] for item in sorted_items]
