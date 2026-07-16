from accelerate import PartialState
from rich import print


def get_state() -> PartialState:
    return PartialState()


def is_main_process() -> bool:
    return get_state().is_main_process


def is_local_main_process() -> bool:
    return get_state().is_local_main_process


def print_main(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def wait_for_everyone() -> None:
    get_state().wait_for_everyone()
