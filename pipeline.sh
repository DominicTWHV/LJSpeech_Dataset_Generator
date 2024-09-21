#!/bin/bash

#update packages
sudo apt update && sudo apt upgrade -y

#reboot?
echo "System updated. You may need to reboot if kernel updates were applied. Reboot now? (y/n)"
read reboot_choice
if [ "$reboot_choice" = "y" ]; then
    sudo reboot
fi

#clone repo
git clone https://github.com/DominicTWHV/LJSpeech_Dataset_Generator.git

#make directories
mkdir -p input
mkdir -p wavs

#install dependencies
sudo apt install -y python3 python3-venv python3-pip ffmpeg zip

#venv setup
python3 -m venv venv
source venv/bin/activate

#install pip packages
pip install -r requirements.txt

#tells user to move files in
echo "Place your .wav files in the 'input/' directory."

#asks to run or not
echo "Ready to run the main.py script to generate the dataset. Run now? (y/n)"
read run_choice
if [ "$run_choice" = "y" ]; then
    python3 main.py
else
    echo "You can run it manually later with: python3 main.py"
fi

#deactivate venv
deactivate

echo "Setup complete!"
