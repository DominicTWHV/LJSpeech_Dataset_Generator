#!/bin/bash

#make directories
mkdir wavs
mkdir output
#grant perms
sudo chown -R "$USER":"$USER" wavs output
chmod -R 755 wavs output

# Check if dependencies are installed
if ! command -v python3 &>/dev/null || \
    ! dpkg -s python3-venv &>/dev/null || \
    ! command -v pip3 &>/dev/null || \
    ! command -v ffmpeg &>/dev/null || \
    ! command -v zip &>/dev/null; then
     sudo apt update
     sudo apt install -y python3 python3-venv python3-pip ffmpeg zip
else
     echo "[OK] Dependencies already installed."
fi

#reboot?
echo
echo -e "\e[35m=======================================================\e[0m"
echo -e "[OK] \e[36mDependencies installed. You may need to reboot if kernel updates were applied. Reboot now? (y/n)\e[0m"
read reboot_choice
if [ "$reboot_choice" = "y" ]; then
    sudo reboot
fi
echo -e "\e[35m=======================================================\e[0m"
echo

#venv setup
python3 -m venv venv
source venv/bin/activate

#install pip packages
pip install -r requirements.txt
echo
echo -e "\e[35m=======================================================\e[0m"
echo
echo -e "\e[36mNotice: Please place your .wav files in the 'wavs/' directory now\e[0m"
echo
echo -e "\e[35m=======================================================\e[0m"
echo
echo -e "\e[36mStart the Gradio WebUI now? (y/n)\e[0m"
read filter_choice
if [ "$filter_choice" = "y" ]; then
    python3 webui.py
else
    echo "[OK] You can run it manually later with: python3 webui.py"
fi
deactivate