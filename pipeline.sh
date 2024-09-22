#!/bin/bash

#update packages
sudo apt update && sudo apt upgrade -y

#reboot?
echo "System updated. You may need to reboot if kernel updates were applied. Reboot now? (y/n)"
read reboot_choice
if [ "$reboot_choice" = "y" ]; then
    sudo reboot
fi

#make directories
mkdir input
mkdir wavs

#grant perms
sudo chown -R "$USER":"$USER" input wavs
chmod -R 755 input wavs

#install dependencies
sudo apt install -y python3 python3-venv python3-pip ffmpeg zip

#venv setup
python3 -m venv venv
source venv/bin/activate

#install pip packages
pip install -r requirements.txt

#tells user to move files in
echo "Notice: Please place your .wav files in the 'input/' directory now"

echo "Do you wish to automatically clean up background noise? (y/n)"
read filter_choice
if [ "$filter_choice" = "y" ]; then
    python3 filter.py
else
    echo "You can run it manually later with: python3 filter.py"
fi

echo "Ready to run the main.py script to generate the dataset? (y/n)"
read run_choice
if [ "$run_choice" = "y" ]; then
    python3 main.py
else
    echo "You can run it manually later with: python3 main.py"
fi

#deactivate venv
deactivate

echo "Setup complete!"
