import os
import pathlib
import time


def env_int(name: str, default_value: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default_value
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got: {v}")


def sleep_if_paused(pause_file: pathlib.Path):
    sleep_index = 0
    while pause_file.exists():
        if not (sleep_index & (sleep_index + 1)):
            print("Sleeping at sleep_index:", sleep_index)
        time.sleep(1)
        sleep_index += 1
    if sleep_index:
        print("Woke up at sleep_index:", sleep_index)
