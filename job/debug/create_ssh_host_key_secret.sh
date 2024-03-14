#!/bin/bash

# initialize the variables
secret="ssh-host-key"

# process the named arguments
while getopts ":s:" opt; do
  case $opt in
    s)
      secret="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# generate ssh host private key & public key
private_key_tempfile="./id_rsa"
public_key_tempfile="./id_rsa.pub"
ssh-keygen -t rsa -b 4096 -N "" -m pem -f $private_key_tempfile
private_key=$(cat "$private_key_tempfile")
public_key=$(cat "$public_key_tempfile")
rm $private_key_tempfile
rm $public_key_tempfile

kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: $secret
  labels:
    tensorstack.dev/owner-name: ssh
data:
  ssh_host_rsa_key: $(echo "$private_key" | base64 -w 0)
  ssh_host_rsa_key.pub: $(echo "$public_key" | base64 -w 0)
type: Opaque
EOF
