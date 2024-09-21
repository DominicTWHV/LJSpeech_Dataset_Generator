#!/bin/bash

#set up dirs
mkdir -p input
mkdir -p wavs

#fetch needed packages
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg zip

echo "Directories created and required packages installed."
