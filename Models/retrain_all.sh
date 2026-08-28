#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"

training_pid=""

cancel_training() {
    printf '\nTraining cancelled. Stopping the current run...\n'
    if [[ -n "$training_pid" ]] && kill -0 "$training_pid" 2>/dev/null; then
        kill -INT "$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
    exit 130
}

run_training() {
    python "$@" &
    training_pid=$!
    wait "$training_pid"
    training_pid=""
}

trap cancel_training INT TERM

run_training tco_SingleAgent_TorchTrain.py
run_training tco_SingleAgent_TorchTrain.py --zhang
run_training tco_SingleAgent_NoTorchTrain.py
run_training tco_SingleAgent_NoTorchTrain.py --zhang

run_training tco_MultiAgent_TorchTrain.py --scenario 2x2
run_training tco_MultiAgent_TorchTrain.py --scenario 2x2 --zhang
run_training tco_MultiAgent_TorchTrain.py --scenario 4x4
run_training tco_MultiAgent_TorchTrain.py --scenario 4x4 --zhang

run_training tco_MultiAgent_NoTorchTrain.py --scenario 2x2
run_training tco_MultiAgent_NoTorchTrain.py --scenario 2x2 --zhang
run_training tco_MultiAgent_NoTorchTrain.py --scenario 4x4
run_training tco_MultiAgent_NoTorchTrain.py --scenario 4x4 --zhang
