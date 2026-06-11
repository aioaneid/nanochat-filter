#!/usr/bin/env bash
set -euo pipefail

FORWARD_FILES=(
    cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu
    cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward.cu
    cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu
)

BACKWARD_FILES=(
    cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu
    cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward.cu
    cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu
)

declare -A FUNC_NAME=(
    ["cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu"]="ltv_look_back_fused_concat_head_major_register_cast_cub_forward_device"
    ["cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_forward.cu"]="ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_forward_device"
    ["cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_forward.cu"]="ltv_fused_concat_head_major_register_cast_cub_forward_device"

    ["cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu"]="ltv_look_back_fused_concat_head_major_register_cast_cub_backward_device"
    ["cpp_ewma/src/ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_cuda_backward.cu"]="ltv_look_back_fused_concat_head_major_register_cast_sequential_scan_backward_device"
    ["cpp_ewma/src/ltv_fused_concat_head_major_register_cast_blelloch_scan_cuda_backward.cu"]="ltv_fused_concat_head_major_register_cast_cub_backward_device"
)

ALL_FILES=("${FORWARD_FILES[@]}" "${BACKWARD_FILES[@]}")

BACKUP_DIR=$(mktemp -d)
for f in "${ALL_FILES[@]}"; do
    cp "$f" "$BACKUP_DIR/$(basename "$f")"
done

cleanup() {
    for f in "${ALL_FILES[@]}"; do
        cp "$BACKUP_DIR/$(basename "$f")" "$f"
    done
    rm -rf "$BACKUP_DIR"
}
trap cleanup EXIT

insert_early_return() {
    local file="$1" func="$2"
    perl - "$file" "$func" <<'EOF'
        my ($file, $func) = @ARGV;
        open(my $fh, '<', $file) or die "cannot open $file: $!";
        my @lines = <$fh>;
        close $fh;

        my $start = -1;
        my $brace = -1;

        for (my $i = 0; $i < @lines; $i++) {
            # Look for the definition start: a line that begins with the function
            # name (after optional whitespace) and an opening parenthesis.
            if ($start == -1) {
                if ($lines[$i] =~ /^\s*\Q$func\E\s*\(/) {
                    $start = $i;
                }
                next;
            }

            # We've found the function signature. Now search for the body's
            # opening brace. If we see a semicolon first, it's a declaration,
            # so we reset and continue looking.
            if ($lines[$i] =~ /^\s*\{/) {
                $brace = $i;
                last;
            }
            if ($lines[$i] =~ /;/) {
                $start = -1;   # not a definition, reset
                last;
            }
        }

        die "could not find opening brace for $func in $file" if $brace == -1;

        # Insert the no-op line right after the opening brace.
        splice @lines, $brace + 1, 0, "    if (true) { return; }\n";

        open(my $out, '>', $file) or die "cannot write $file: $!";
        print $out @lines;
        close $out;
EOF
}

DATE_STAMP=$(date +%F_%H-%M-%S)

# --- Forward no‑op benchmark ---
echo "Patching forward kernels with early return..."
for f in "${FORWARD_FILES[@]}"; do
    insert_early_return "$f" "${FUNC_NAME[$f]}"
done

echo "Running backward (noop-forward) benchmark..."
modal run scripts/modal_app.py::run_perf_base_train --max-forward-specs=1 --max-backward-specs=1000 |&
    tee -a $HOME/tmp/perf-L40S-look-back-noop-forward-1-1000-${DATE_STAMP}.log

echo "Restoring forward kernels..."
for f in "${FORWARD_FILES[@]}"; do
    cp "$BACKUP_DIR/$(basename "$f")" "$f"
done

# --- Backward no‑op benchmark ---
echo "Patching backward kernels with early return..."
for f in "${BACKWARD_FILES[@]}"; do
    insert_early_return "$f" "${FUNC_NAME[$f]}"
done

echo "Running forward (noop-backward) benchmark..."
modal run scripts/modal_app.py::run_perf_base_train --max-forward-specs=1000 --max-backward-specs=1 |&
    tee -a $HOME/tmp/perf-L40S-look-back-noop-backward-1000-1-${DATE_STAMP}.log

echo "Restoring backward kernels..."
for f in "${BACKWARD_FILES[@]}"; do
    cp "$BACKUP_DIR/$(basename "$f")" "$f"
done

echo "Done."
