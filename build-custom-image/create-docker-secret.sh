#!/bin/bash

# initialize the variables
secret="docker-config"
registry=""
username=""
password=""

# process the named arguments
while getopts ":s:r:u:p:" opt; do
  case $opt in
    s)
      secret="$OPTARG"
      ;;
    r)
      registry="$OPTARG"
      ;;
    u)
      username="$OPTARG"
      ;;
    p)
      password="$OPTARG"
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

# check if registry is missing
if [[ -z "$registry" ]]; then
  echo "Usage: $0 -r <registry> [-u <username>] [-p <password>] [-s <secret>]" >&2
  exit 1
fi

# check if username is missing
if [[ -z "$username" ]]; then
  read -p "Enter username: " username
fi

# check if password is missing
if [[ -z "$password" ]]; then
  read -s -p "Enter password: " password
  echo ""
fi

# print the arguments
echo "Secret: $secret"
echo "Registry: $registry"
echo "Username: $username"

auth=$(echo -n $username:$password | base64 -w 0)
configJSON='{"auths":{"'$registry'/v1/":{"username":"'$username'","password":"'$password'","auth":"'$auth'"}}}'

kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: $secret
  labels:
    tensorstack.dev/resource: docker
data:
  .dockerconfigjson: $(echo $configJSON | base64 -w 0)
type: Opaque
EOF
