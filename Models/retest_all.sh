#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"

testing_pid=""

cancel_testing() {
    printf '\nTesting cancelled. Stopping the current run...\n'
    if [[ -n "$testing_pid" ]] && kill -0 "$testing_pid" 2>/dev/null; then
        kill -INT "$testing_pid" 2>/dev/null || true
        wait "$testing_pid" 2>/dev/null || true
    fi
    exit 130
}

run_testing() {
    python "$@" &
    testing_pid=$!
    wait "$testing_pid"
    testing_pid=""
}

trap cancel_testing INT TERM

run_testing tco_SingleAgent_TorchTest.py
run_testing tco_SingleAgent_TorchTest.py --zhang
run_testing tco_SingleAgent_NoTorchTest.py
run_testing tco_SingleAgent_NoTorchTest.py --zhang

run_testing tco_MultiAgent_TorchTest.py --scenario 2x2
run_testing tco_MultiAgent_TorchTest.py --scenario 2x2 --zhang
run_testing tco_MultiAgent_TorchTest.py --scenario 4x4
run_testing tco_MultiAgent_TorchTest.py --scenario 4x4 --zhang

run_testing tco_MultiAgent_NoTorchTest.py --scenario 2x2
run_testing tco_MultiAgent_NoTorchTest.py --scenario 2x2 --zhang
run_testing tco_MultiAgent_NoTorchTest.py --scenario 4x4
run_testing tco_MultiAgent_NoTorchTest.py --scenario 4x4 --zhang
