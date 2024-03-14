#!/bin/bash

# initialize the variables
secret="ssh-user-public-key"
key_file="~/.ssh/id_rsa.pub"

# process the named arguments
while getopts ":s:f:" opt; do
  case $opt in
    s)
      secret="$OPTARG"
      ;;
    f)
      key_file="$OPTARG"
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
public_key=$(cat "$key_file")

kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: $secret
  labels:
    tensorstack.dev/resource: ssh
data:
  ssh-publickey: $(echo "$public_key" | base64 -w 0)
type: Opaque
EOF
