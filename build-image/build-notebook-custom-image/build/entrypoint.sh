#!/bin/bash

# start notebook
sh -c "jupyter notebook --notebook-dir=/t9k/mnt --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}" 2>&1 &
# start ssh server
if [[ "${ENABLE_SSH_SERVER}" == "true" ]]; then
  echo "Generating ssh key and start ssh server ..." >&2
  sh -c "/t9k/app/ssh_server.sh" 2>&1  &
fi

# just keep this script running
while [[ true ]]; do
    sleep 1
done
