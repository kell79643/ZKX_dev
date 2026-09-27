#!/usr/bin/env bash
set -eu
exec bash "$(dirname "$0")/../../../shared/radartools/run_case.sh" ca_cfar "$@"
