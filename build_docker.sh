#!/usr/bin/env bash

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root" >&2
if command -v docker >/dev/null 2>&1; then
    echo "Docker is already installed"
else
    echo "Docker could not be found, installing..." >&2
    
    apt update
    apt install -y docker.io
fi
    apt install -y docker.io
fi

docker build -t ljspeech_dsg .

docker run ljspeech_dsg