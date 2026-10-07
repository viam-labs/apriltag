#!/bin/sh
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME=".venv"
PYTHON="$VENV_NAME/bin/python"
ENV_ERROR="This module requires Python >=3.10, pip, and virtualenv to be installed."

if ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
    echo "Failed to create virtualenv."
    if command -v apt-get >/dev/null; then
        echo "Detected Debian/Ubuntu, attempting to install python3-venv automatically."
        SUDO="sudo"
        if ! command -v $SUDO >/dev/null; then
            SUDO=""
        fi
		if ! apt info python3-venv >/dev/null 2>&1; then
			echo "Package info not found, trying apt update"
			$SUDO apt -qq update >/dev/null
		fi
        $SUDO apt install -qqy python3-venv >/dev/null 2>&1
        if ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
            echo $ENV_ERROR >&2
            exit 1
        fi
    else
        echo $ENV_ERROR >&2
        exit 1
    fi
fi

# -qq suppresses extraneous output from pip. We intentionally do NOT pass -U:
# with -U pip re-resolves and upgrades every dependency from the network on
# every module start, which slows restarts and can trip viam-server's config
# validation deadline. Without -U, already-satisfied requirements are a no-op,
# so restarts (and hot reloads) are fast and work offline once installed.
echo "Virtualenv found/created. Installing Python packages..."
if ! $PYTHON -m pip install -r requirements.txt -qq; then
    exit 1
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
echo "Starting module..."
exec $PYTHON -m src.main $@