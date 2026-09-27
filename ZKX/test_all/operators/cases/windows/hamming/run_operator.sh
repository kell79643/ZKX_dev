#!/usr/bin/env bash
set -eu
exec bash "${0%/*}/../../../shared/filtering/run_case.sh" hamming "$@"
