#!/usr/bin/env bash
# Pipes stdin as a script to run on the given vast instance.

if [[ -z $1 ]]; then
  echo "Usage: ./ssh_vast_instance.sh VAST_ID"
  exit 1
fi

instance=$(vast show instances | grep $1)
if [[ $(echo "$instance" | awk '{ print $3 }') != "running" ]]; then
  echo "Vast instance '$1' cannot be found, or isn't running."
  exit 1
fi

host=$(echo "$instance" | awk '{ print $10 }')
port=$(echo "$instance" | awk '{ print $11 }')
ssh -p $port root@$host /bin/bash < /dev/stdin
