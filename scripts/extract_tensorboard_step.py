import os
import glob
import argparse
import csv
from distutils.util import strtobool
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_event_file(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def find_event(events, target_step, exact=False):
    if not events:
        return None

    if exact:
        for e in events:
            if e.step == target_step:
                return e
        return None

    # nearest step
    return min(events, key=lambda e: abs(e.step - target_step))


def extract_step_data(log, target_step, exact: bool):
    event_files = (
        glob.glob(
            os.path.join(log, "**", "events.out.tfevents*"),
            recursive=True,
        )
        if os.path.isdir(log)
        else [log]
    )

    results = defaultdict(dict)

    for path in event_files:
        try:
            ea = load_event_file(path)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        scalar_tags = ea.Tags().get("scalars", [])

        for tag in scalar_tags:
            events = ea.Scalars(tag)
            event = find_event(events, target_step, exact=exact)
            if event:
                results[tag][path] = {
                    "step": event.step,
                    "value": event.value,
                }

    return results


def print_results(results, target_step):
    print(f"\n===== Scalars near step {target_step} =====\n")

    for tag in sorted(results.keys()):
        print(f"\nTag: {tag}")
        for path, info in results[tag].items():
            fname = os.path.basename(path)
            print(f"  File: {fname}")
            print(f"    Step:  {info['step']}")
            print(f"    Value: {info['value']:.6f}")


def aggregate_results(results):
    import numpy as np

    print("\n===== Aggregated (mean/std across files) =====\n")

    for tag in sorted(results.keys()):
        values = [info["value"] for info in results[tag].values()]
        if not values:
            continue

        mean = np.mean(values)
        std = np.std(values)
        print(f"{tag:40s} mean={mean:.6f} std={std:.6f} n={len(values)}")


def export_csv(results, output_path):
    rows = []
    for tag, files in results.items():
        for path, info in files.items():
            rows.append(
                {
                    "tag": tag,
                    "file": os.path.basename(path),
                    "step": info["step"],
                    "value": info["value"],
                }
            )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "file", "step", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] CSV exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract scalar values at (or nearest to) a given step from TensorBoard event files."
    )
    parser.add_argument(
        "--log", required=True, help="Directory or file containing TensorBoard logs"
    )
    parser.add_argument("--step", type=int, required=True, help="Target global step")
    parser.add_argument(
        "--exact",
        type=lambda v: bool(strtobool(v)),
        default=False,
        help="Print mean/std per tag across files",
    )
    parser.add_argument(
        "--aggregate",
        type=lambda v: bool(strtobool(v)),
        default=False,
        help="Print mean/std per tag across files",
    )
    parser.add_argument(
        "--csv", type=str, help="Optional path to export results as CSV"
    )

    args = parser.parse_args()

    results = extract_step_data(args.log, args.step, exact=args.exact)

    print_results(results, args.step)

    if args.aggregate:
        aggregate_results(results)

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()
