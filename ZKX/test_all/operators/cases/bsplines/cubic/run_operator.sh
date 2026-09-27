#!/usr/bin/env bash
set -eu
exec bash "${0%/*}/../../../suites/task2_final/run_case.sh" cubic "$@"
