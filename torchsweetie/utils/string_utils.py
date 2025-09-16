def is_chinese(c: str) -> bool:
    assert len(c) == 1

    if "\u4e00" <= c <= "\u9fa5":
        return True
    else:
        return False


def display_len(string: str) -> int:
    length = 0
    for c in string:
        if is_chinese(c):
            length += 2
        else:
            length += 1

    return length


def format_string(string: str, max_length: int) -> str:
    length = display_len(string)

    while length < max_length:
        string = " " + string
        length += 1

    return string
