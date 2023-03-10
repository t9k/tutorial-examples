registry=$1
username=$2
password=$3

auth=$(echo -n $username:$password | base64 -w 0)
configJSON='{"auths":{"'$registry'/v1/":{"username":"'$username'","password":"'$password'","auth":"'$auth'"}}}'

kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: docker-config
data:
  config.json: $(echo $configJSON | base64 -w 0)
type: Opaque
EOF
