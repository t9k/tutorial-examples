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

# check if any arguments are missing
if [[ -z "$secret" || -z "$registry" || -z "$username" || -z "$password" ]]; then
  echo "Usage: $0 -r <registry> -u <username> -p <password> [-s <secret>]" >&2
  exit 1
fi

# print the arguments
echo "Secret: $secret"
echo "Registry: $registry"
echo "Username: $username"
echo "Password: $password"

auth=$(echo -n $username:$password | base64 -w 0)
configJSON='{"auths":{"'$registry'/v1/":{"username":"'$username'","password":"'$password'","auth":"'$auth'"}}}'

kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: $secret
data:
  config.json: $(echo $configJSON | base64 -w 0)
type: Opaque
EOF
