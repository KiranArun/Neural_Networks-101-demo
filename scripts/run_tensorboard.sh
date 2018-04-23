#!/bin/bash

echo "Running tensorboard..."

tensorboard --logdir ./logdir/ --host 0.0.0.0 --port 6006 & true;
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip > /dev/null 2>&1
unzip -o ngrok-stable-linux-amd64.zip > /dev/null 2>&1
./ngrok http 6006 & true;
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"