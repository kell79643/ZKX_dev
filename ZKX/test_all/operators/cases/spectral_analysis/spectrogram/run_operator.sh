#!/usr/bin/env bash
set -eu
exec bash "${0%/*}/../../../suites/task2_features/run_case.sh" spectrogram "$@"
