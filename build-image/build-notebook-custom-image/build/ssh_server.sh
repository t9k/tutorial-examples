SSH_PATH="/t9k/app/.ssh"
mkdir ${SSH_PATH}

awk '{print}' /t9k/authorized_keys/* > ${SSH_PATH}/authorized_keys

for key in $(ls /t9k/host_keys);do
  cat /t9k/host_keys/$key > ${SSH_PATH}/$key
  chmod 0600 ${SSH_PATH}/$key
done

echo "Port 2222
HostKey ${SSH_PATH}/ssh_host_rsa_key
PubkeyAuthentication yes
AuthorizedKeysFile  ${SSH_PATH}/authorized_keys
ChallengeResponseAuthentication no
Subsystem   sftp    internal-sftp
PidFile ${SSH_PATH}/sshd.pid" > ${SSH_PATH}/sshd_config

chmod 0700 ${SSH_PATH}
chmod 0600 ${SSH_PATH}/authorized_keys
chmod 0600 ${SSH_PATH}/sshd_config

/usr/sbin/sshd -f ${SSH_PATH}/sshd_config -D
