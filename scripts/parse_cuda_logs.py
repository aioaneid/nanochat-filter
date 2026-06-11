import re
import sys
import pandas as pd

# -----------------------------------------------------------------------------
# Regexes
# -----------------------------------------------------------------------------

kernel_re = re.compile(
    r"Function properties for (\S+)"
)

stats_re = re.compile(
    r"(\d+) bytes stack frame,\s*"
    r"(\d+) bytes spill stores,\s*"
    r"(\d+) bytes spill loads"
)

# Matches:
# Li128ELi8ELi64ELi2
#
# Interpreted as:
#   k_block_threads
#   k_items_per_thread
#   k_r_blocks
#   k_r_items_per_thread
dispatch_re = re.compile(
    r"Li(\d+)ELi(\d+)ELi(\d+)ELi(\d+)"
)

# -----------------------------------------------------------------------------
# Parse
# -----------------------------------------------------------------------------

rows = []

lines = sys.stdin.read().splitlines()

i = 0
while i < len(lines):
    line = lines[i]

    kernel_match = kernel_re.search(line)

    if kernel_match:
        kernel_name = kernel_match.group(1)

        dispatch_match = dispatch_re.search(kernel_name)

        if dispatch_match:
            k_block_threads = int(dispatch_match.group(1))
            k_items_per_thread = int(dispatch_match.group(2))
            k_r_blocks = int(dispatch_match.group(3))
            k_r_items_per_thread = int(dispatch_match.group(4))
        else:
            k_block_threads = None
            k_items_per_thread = None
            k_r_blocks = None
            k_r_items_per_thread = None

        # Next line should contain stats
        if i + 1 < len(lines):
            stats_line = lines[i + 1]

            stats_match = stats_re.search(stats_line)

            if stats_match:
                stack_frame = int(stats_match.group(1))
                spill_stores = int(stats_match.group(2))
                spill_loads = int(stats_match.group(3))

                total_spills = spill_stores + spill_loads

                rows.append(
                    {
                        "kernel_name": kernel_name,
                        "k_block_threads": k_block_threads,
                        "k_items_per_thread": k_items_per_thread,
                        "k_r_blocks": k_r_blocks,
                        "k_r_items_per_thread": k_r_items_per_thread,
                        "stack_frame_bytes": stack_frame,
                        "spill_stores_bytes": spill_stores,
                        "spill_loads_bytes": spill_loads,
                        "total_spill_bytes": total_spills,
                    }
                )

        i += 2
    else:
        i += 1

# -----------------------------------------------------------------------------
# DataFrame
# -----------------------------------------------------------------------------

df = pd.DataFrame(rows)

# Optional derived metrics
df["threads_x_items"] = (
    df["k_block_threads"] * df["k_items_per_thread"]
)

df["r_threads_x_items"] = (
    df["k_r_blocks"] * df["k_r_items_per_thread"]
)

# Sort by spill severity
df = df.sort_values(
    by=[
        "total_spill_bytes",
        "stack_frame_bytes",
    ],
    ascending=False,
)

# -----------------------------------------------------------------------------
# Pretty formatting
# -----------------------------------------------------------------------------

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

summary_cols = [
    "k_block_threads",
    "k_items_per_thread",
    "k_r_blocks",
    "k_r_items_per_thread",
    "stack_frame_bytes",
    "spill_stores_bytes",
    "spill_loads_bytes",
    "total_spill_bytes",
]

print("\n=== Full Table ===\n")
print(df[summary_cols].to_string(index=False))

print("\n=== Best Configurations (lowest spills) ===\n")
print(
    df.sort_values(
        by=["total_spill_bytes", "stack_frame_bytes"]
    )[summary_cols]
    .head(20)
    .to_string(index=False)
)

print("\n=== Worst Configurations (highest spills) ===\n")
print(
    df.sort_values(
        by=["total_spill_bytes", "stack_frame_bytes"],
        ascending=False,
    )[summary_cols]
    .head(20)
    .to_string(index=False)
)
