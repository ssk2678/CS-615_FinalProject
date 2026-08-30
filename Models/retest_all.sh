#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
EVAL_SEED="${EVAL_SEED:-42}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/retest_results/$RUN_TIMESTAMP"
LOG_DIR="$RESULTS_DIR/logs"
RESULTS_TSV="$RESULTS_DIR/comparison.tsv"
COMPARISON_CSV="$RESULTS_DIR/comparison.csv"

mkdir -p "$LOG_DIR"
: > "$RESULTS_TSV"

if ! command -v python >/dev/null 2>&1; then
    printf 'python was not found. Activate the cs615-sumo Conda environment first.\n' >&2
    exit 1
fi

if ! python -c 'import numpy, torch, sumo_rl' >/dev/null 2>&1; then
    printf 'Required Python packages are unavailable. Activate the cs615-sumo Conda environment first.\n' >&2
    exit 1
fi

cancel_run() {
    printf '\nRetest cancelled. Completed logs remain in:\n%s\n' "$RESULTS_DIR"
    exit 130
}

trap cancel_run INT TERM

failure_count=0

extract_value() {
    local label="$1"
    local log_file="$2"

    sed -nE "s/^${label}:[[:space:]]*(-?[+]?[0-9]+([.][0-9]+)?).*/\\1/p" \
        "$log_file" | tail -n 1
}

run_test() {
    local key="$1"
    local label="$2"
    shift 2

    local log_file="$LOG_DIR/$key.log"
    local exit_status
    local before
    local after
    local delta
    local result

    printf '\n============================================================\n'
    printf '%s\n' "$label"
    printf 'Command: python %q' "$1"
    printf ' %q' "${@:2}"
    printf '\n'

    set +e
    python "$@" 2>&1 | tee "$log_file"
    exit_status=${PIPESTATUS[0]}
    set -e

    before="$(extract_value 'Before \(Random\) Reward' "$log_file")"
    after="$(extract_value 'After \(Trained\) Reward' "$log_file")"
    result="$(sed -nE 's/^Result:[[:space:]]*(IMPROVED|WORSE|SAME).*/\1/p' "$log_file" | tail -n 1)"

    if (( exit_status != 0 )) || [[ -z "$before" || -z "$after" || -z "$result" ]]; then
        before="${before:-ERROR}"
        after="${after:-ERROR}"
        delta="N/A"
        result="INCOMPLETE"
        failure_count=$((failure_count + 1))
        printf 'Test did not complete; see %s\n' "$log_file" >&2
    else
        delta="$(awk -v before="$before" -v after="$after" \
            'BEGIN { printf "%+.2f", after - before }')"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$key" "$label" "$before" "$after" "$delta" "$result" >> "$RESULTS_TSV"
}

print_comparison() {
    local key
    local label
    local before
    local after
    local delta
    local result

    printf '\nRandom-before / trained-after evaluation (seed %s)\n' "$EVAL_SEED"
    printf '%-34s %12s %12s %12s %-10s\n' \
        'Configuration' 'Before' 'After' 'Delta' 'Result'
    printf '%-34s %12s %12s %12s %-10s\n' \
        '----------------------------------' '------------' '------------' '------------' '----------'

    printf 'Configuration,Before Random,After Trained,Delta,Result\n' > "$COMPARISON_CSV"
    while IFS=$'\t' read -r key label before after delta result; do
        printf '%-34s %12s %12s %12s %-10s\n' \
            "$label" "$before" "$after" "$delta" "$result"
        printf '"%s",%s,%s,%s,%s\n' \
            "$label" "$before" "$after" "$delta" "$result" >> "$COMPARISON_CSV"
    done < "$RESULTS_TSV"

    printf '\nHigher reward is better within the same configuration.\n'
    printf 'Do not compare Standard directly with Zhang-style; their reward definitions differ.\n'
    printf 'Logs and summary CSV:\n%s\n' "$RESULTS_DIR"
}

printf 'Evaluation seed: %s\n' "$EVAL_SEED"
printf 'Results directory: %s\n' "$RESULTS_DIR"
printf 'Each test evaluates a random network, then loads and evaluates its existing trained weights.\n'
printf 'No training scripts are run and no model weights are modified.\n'

run_test single_torch_standard 'Single | PyTorch | Standard' \
    tco_SingleAgent_TorchTest.py --seed "$EVAL_SEED"
run_test single_torch_zhang 'Single | PyTorch | Zhang-style' \
    tco_SingleAgent_TorchTest.py --zhang --seed "$EVAL_SEED"
run_test single_numpy_standard 'Single | NumPy | Standard' \
    tco_SingleAgent_NoTorchTest.py --seed "$EVAL_SEED"
run_test single_numpy_zhang 'Single | NumPy | Zhang-style' \
    tco_SingleAgent_NoTorchTest.py --zhang --seed "$EVAL_SEED"

run_test 2x2_torch_standard '2x2 | PyTorch | Standard' \
    tco_MultiAgent_TorchTest.py --scenario 2x2 --seed "$EVAL_SEED"
run_test 2x2_torch_zhang '2x2 | PyTorch | Zhang-style' \
    tco_MultiAgent_TorchTest.py --scenario 2x2 --zhang --seed "$EVAL_SEED"
run_test 2x2_numpy_standard '2x2 | NumPy | Standard' \
    tco_MultiAgent_NoTorchTest.py --scenario 2x2 --seed "$EVAL_SEED"
run_test 2x2_numpy_zhang '2x2 | NumPy | Zhang-style' \
    tco_MultiAgent_NoTorchTest.py --scenario 2x2 --zhang --seed "$EVAL_SEED"

run_test 4x4_torch_standard '4x4 | PyTorch | Standard' \
    tco_MultiAgent_TorchTest.py --scenario 4x4 --seed "$EVAL_SEED"
run_test 4x4_torch_zhang '4x4 | PyTorch | Zhang-style' \
    tco_MultiAgent_TorchTest.py --scenario 4x4 --zhang --seed "$EVAL_SEED"
run_test 4x4_numpy_standard '4x4 | NumPy | Standard' \
    tco_MultiAgent_NoTorchTest.py --scenario 4x4 --seed "$EVAL_SEED"
run_test 4x4_numpy_zhang '4x4 | NumPy | Zhang-style' \
    tco_MultiAgent_NoTorchTest.py --scenario 4x4 --zhang --seed "$EVAL_SEED"

print_comparison

if (( failure_count > 0 )); then
    printf '\n%s test configuration(s) did not complete.\n' "$failure_count" >&2
    exit 1
fi
