import pathlib
import re
import argparse
import sys
from typing import Iterable
import pandas as pd
from pathlib import Path


def parse_nanochat_logs(
    file_paths: Iterable[pathlib.Path], min_step: int, max_steps: int
) -> pd.DataFrame:
    # Regex for the command line config
    config_pattern = re.compile(
        r"python -m scripts\.base_train"
        r".*"
        r"\s+--depth=(?P<depth>\d+)"
        r".*"
        r"\s+--model_type=(?P<m_type>[\w_]+)"
        r".*?"
        r"(?:\s+--ltv_r=(?P<r>\d+))?"
        r"\s+--ltv_query=(?P<q>none|active)"
        r"\s+--ltv_key=(?P<k>none|active)"
        r"\s+--ltv_value=(?P<v>none|active)"
        r"(?:\s+--alpha_mlp_bias_init=(?P<ambi>[-\d.]+))?"
        r"(?:\s+--rand_filter_bias=(?P<rfb>true|false))?"
        r"(?:\s+--rand_filter_init=(?P<rfi>true|false))?"
    )

    # Regex for metrics
    metrics_pattern = re.compile(
        r"step (?P<step>\d+)/\d+.*?loss: (?P<loss>[\d.]+).*?"
        r"grad norm: (?P<grad_norm>[\d.]+).*?"
        r"dt: (?P<dt>[\d.]+)ms.*?"
        r"tok/sec: (?P<tok_sec>[\d,]+)"
    )

    all_runs = []
    current_run = None

    summary_schema = {
        "Configuration": "str",
        "Step Range": "str",
        "min(Loss)": "float64",
        "last(Loss)": "float64",
        "avg(norm(Grad))": "float64",
        "std(norm(Grad))": "float64",
        "max(tok/sec)": "int64",
        "avg(tok/sec)": "float64",
        "max(Loss)": "float64",
    }

    for file_path in file_paths:
        with open(file_path, "r") as f:
            for line in f:
                # New section detector (command line)
                config_match = config_pattern.search(line)
                if config_match:
                    if current_run and current_run["data"]:
                        all_runs.append(current_run)

                    cfg = config_match.groupdict()
                    label = f"{cfg['m_type']}(d={cfg['depth']}) [R:{cfg['r']} Q:{cfg['q']} K:{cfg['k']} V:{cfg['v']} Ambi:{cfg['ambi']} Rfb:{cfg['rfb']} Rfi:{cfg['rfi']}]"
                    current_run = {"label": label, "data": []}
                    continue

                # Metric line detector
                metric_match = metrics_pattern.search(line)
                if metric_match and current_run:
                    d = metric_match.groupdict()
                    if (step := int(d["step"])) >= max_steps:
                        continue
                    if step < min_step:
                        continue
                    current_run["data"].append(
                        {
                            "step": step,
                            "loss": float(d["loss"]),
                            "grad_norm": float(d["grad_norm"]),
                            "dt": float(d["dt"]),
                            "tok_sec": int(d["tok_sec"].replace(",", "")),
                        }
                    )

    if current_run and current_run["data"]:
        all_runs.append(current_run)

    summary_rows = []
    for run in all_runs:
        df = pd.DataFrame(run["data"])

        # For timing stats, ignore the first 2 steps of the group due to compilation,
        # and also those which are about 2x slower, which is the case on Low Power.
        stable_df = df.tail(max(len(df) - 2, 1))
        stable_df = stable_df[stable_df["dt"] < stable_df["dt"].min() * 1.75]

        # Step range logic
        start_step = df["step"].min()
        end_step = df["step"].max() + 1

        summary_rows.append(
            {
                "Configuration": run["label"],
                "Step Range": f"[{start_step}, {end_step})",
                "min(Loss)": df["loss"].min(),
                "last(Loss)": df["loss"].iloc[-1],
                "avg(norm(Grad))": round(df["grad_norm"].mean(), 4),
                "std(norm(Grad))": round(df["grad_norm"].std(ddof=0), 4),
                "max(tok/sec)": int(stable_df["tok_sec"].max()),
                "avg(tok/sec)": int(stable_df["tok_sec"].mean()),
                "max(Loss)": df["loss"].max(),
            }
        )

    return pd.DataFrame(summary_rows).astype(summary_schema)


def main():
    parser = argparse.ArgumentParser(description="Analyze Nanochat LTV speedrun logs.")
    parser.add_argument(
        "--logfile", nargs="*", help="Path to the training log file", type=pathlib.Path
    )
    parser.add_argument(
        "--sort", choices=["loss", "speed"], help="Sort by loss or step speed"
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=sys.maxsize,
        help="Path to the training log file",
    )

    args = parser.parse_args()

    df = parse_nanochat_logs(args.logfile, args.min_step, args.max_steps)

    if args.sort:
        sort_col = "Min Loss" if args.sort == "loss" else "ms/step"
        df = df.sort_values(by=sort_col, ascending=(args.sort == "loss"))

    # Group by Step Range and print a table for each
    for step_range, group in df.groupby("Step Range", sort=False):
        print(f"\nRange: {step_range}")
        print(group.drop(columns=["Step Range"]).to_string(index=False))
        print("-" * 40)


if __name__ == "__main__":
    main()
