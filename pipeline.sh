#!/bin/bash

#update packages
sudo apt update && sudo apt upgrade -y

#make directories
mkdir wavs
mkdir output
#grant perms
sudo chown -R "$USER":"$USER" wavs output
chmod -R 755 wavs output

#install dependencies
sudo apt install -y python3 python3-venv python3-pip ffmpeg zip

#reboot?
echo "[OK] System updated. You may need to reboot if kernel updates were applied. Reboot now? (y/n)"
read reboot_choice
if [ "$reboot_choice" = "y" ]; then
    sudo reboot
fi

#venv setup
python3 -m venv venv
source venv/bin/activate

#install pip packages
pip install -r requirements.txt
echo
echo
echo "======================================================="
echo
echo "Notice: Please place your .wav files in the 'wavs/' directory now"
echo
echo "======================================================="
echo
echo "Do you wish to automatically clean up background noise? Please note this function is a little quirky. Use at your own risk. (y/n)"
read filter_choice
if [ "$filter_choice" = "y" ]; then
    python3 filter.py
else
    echo "[OK] You can run it manually later with: python3 filter.py"
fi
echo
echo "======================================================="
echo
echo "Do you wish to split the audio files into smaller chunks? It's recommended that you do so. (y/n)"
read filter_choice
if [ "$filter_choice" = "y" ]; then
    python3 split.py
else
    echo "[OK] You can run it manually later with: python3 split.py"
fi
echo
echo "======================================================="
echo
echo "Ready to run the main.py script to generate the dataset? (y/n)"
read run_choice
if [ "$run_choice" = "y" ]; then
    python3 main.py
else
    echo "[OK] You can run it manually later with: python3 main.py"
fi
echo
echo "======================================================="
echo
#deactivate venv
deactivate

echo "[OK] Dataset Generation Complete! You should now see a zip folder located in output/ named dataset.zip"
