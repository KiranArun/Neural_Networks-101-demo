#!/bin/bash

echo "Running tensorboard..."

python3 /content/Neural_Networks-demo/scripts/run_tensorboard_command.py

if ! [ -e /content/ngrok ]
then
    wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip > /dev/null 2>&1
    unzip -o ngrok-stable-linux-amd64.zip > /dev/null 2>&1
fi

/content/ngrok http 6006 &

curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print('Tensorboard Link:', json.load(sys.stdin)['tunnels'][0]['public_url'])"