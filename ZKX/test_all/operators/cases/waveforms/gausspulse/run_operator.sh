#!/usr/bin/env bash
set -eu
exec bash "$(dirname "$0")/../../../shared/waveforms/run_case.sh" gausspulse "$@"
