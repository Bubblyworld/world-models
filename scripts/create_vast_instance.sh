#!/usr/bin/env bash
# Deploys and installs the repo on an instance of the given vast.ai type.

if [[ -z $1 ]]; then
  echo "Usage: ./create_vast_instance.sh VAST_TYPE"
  exit 1
fi

echo "Creating instance of type '$1'..."
info=`vast create instance $1\
  --image tensorflow/tensorflow\
  --label latest-gpu --disk 32 | tail -c +10`

if [[ $info == *"available"* ]]; then
  echo "...but instance type '$1' is not available."
  exit 1
fi

id=$(python -c "x=$info; print(x['new_contract'])")
echo "...done, instance id is '$id'."

echo
echo "Waiting for instance to finish initialising..."
sleep 5 # give vast api some time to think
while [[ $(vast show instances | grep $id | awk '{ print $3 }') != "running" ]]; do
  echo "..."
  sleep 5
done
instance=$(vast show instances | grep $id)
echo "...done!"

echo
echo "Copying repo over to new host..."
host=$(echo "$instance" | awk '{ print $10 }')
port=$(echo "$instance" | awk '{ print $11 }')
tar czf - $(dirname $(dirname $0)) | ssh -o StrictHostKeyChecking=accept-new -p $port root@$host "mkdir /root/world-models && tar -xvz -C /root/world-models"
echo "...done!"

echo
echo "Running installation scripts..."
ssh -p $port root@$host "pip install poetry && cd /root/world-models && poetry install"
echo "Done! Instance is ready for abuse."
