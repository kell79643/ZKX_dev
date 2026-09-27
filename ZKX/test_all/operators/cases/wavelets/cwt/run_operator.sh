#!/usr/bin/env bash
set -eu
exec bash "$(dirname "$0")/../../../shared/wavelets/run_case.sh" cwt "$@"
