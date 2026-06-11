from torch.utils.tensorboard import SummaryWriter

import ndjson
import pathlib
import numbers

from nanochat.common import get_base_dir


def _flatten_metrics(metrics: dict, parent_key=""):
    """
    Recursively flatten metrics.
    - Numbers -> scalar
    - Strings -> text
    - Dicts -> recursive
    - Lists/Tuples -> indexed recursively with zero-padded keys
    """
    flat = {}
    for k, v in metrics.items():
        new_key = f"{parent_key}/{k}" if parent_key else k

        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, new_key))
        elif isinstance(v, (list, tuple)):
            n_digits = len(str(len(v) - 1))
            indexed_dict = {str(i).zfill(n_digits): val for i, val in enumerate(v)}
            flat.update(_flatten_metrics(indexed_dict, new_key))
        elif isinstance(v, numbers.Number):
            flat[new_key] = ("scalar", v)
        elif isinstance(v, str):
            flat[new_key] = ("text", v)
        elif isinstance(v, bool):
            flat[new_key] = ("scalar", int(v))
        else:
            # ignore unsupported types
            pass
    return flat


class TensorboardAdapter:
    def __init__(self, project, name, config, save_code=False):
        log_path = (
            pathlib.PosixPath(get_base_dir())
            .joinpath("logs")
            .joinpath(project)
            .joinpath(name)
            .expanduser()
        )
        log_path.mkdir(parents=True, exist_ok=True)
        with open(log_path.joinpath("config.jsonl"), mode="at") as fp:
            writer = ndjson.writer(fp)
            writer.writerow(config)
        self.writer = SummaryWriter(log_dir=log_path)
        self.step = 0

    def log(self, metrics: dict):
        step = metrics.pop("step")
        flat_metrics = _flatten_metrics(metrics)
        for k, (typ, v) in flat_metrics.items():
            if typ == "scalar":
                self.writer.add_scalar(k, v, step)
            elif typ == "text":
                self.writer.add_text(k, v, step)

    def finish(self):
        self.writer.close()
