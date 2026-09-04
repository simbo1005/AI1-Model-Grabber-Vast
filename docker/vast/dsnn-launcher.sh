#!/bin/bash
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_serverless.sh"
. "${utils}/exit_portal.sh" "Workflow Downloader"

. /venv/main/bin/activate

export PYTHONPATH="/opt/dsnn${PYTHONPATH:+:${PYTHONPATH}}"
cd /opt/dsnn

exec python -m launcher.bootstrap
