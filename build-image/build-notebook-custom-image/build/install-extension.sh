#!/usr/bin/env bash

# Bash 'Strict Mode'
# http://redsymbol.net/articles/unofficial-bash-strict-mode

set -euo pipefail

main() {
  # # install notebook2workflow
  # pip3 install --proxy= -i https://pypi.tuna.tsinghua.edu.cn/simple /tmp/notebook2workflow-0.1.0-py3-none-any.whl

  # install jupyter-tensorboard-pro
  pip3 install --proxy= -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab-tensorboard-pro

  # do NOT install jupyterlab-s3-browser for now since it is not capable of listing only the buckets that a user have access to
  # pip3 install --proxy= -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab-s3-browser
  # jupyter serverextension enable --py jupyterlab_s3_browser

  # clean cache
  pip3 cache purge
  # jlpm cache clean
  jupyter lab clean -y
}

main "${@:-}"
