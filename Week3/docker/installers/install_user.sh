#!/usr/bin/env bash

# Fail on first error.
set -e

USER_NAME=josep

adduser --disabled-password --gecos '' ${USER_NAME}
usermod -aG sudo ${USER_NAME}
echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

echo """
ulimit -c unlimited
export GIT_SSL_NO_VERIFY=1
""" >> /home/${USER_NAME}/.bashrc

