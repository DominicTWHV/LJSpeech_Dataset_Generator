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
echo
echo -e "\e[35m=======================================================\e[0m"
echo -e "[OK] \e[36mSystem updated. You may need to reboot if kernel updates were applied. Reboot now? (y/n)\e[0m"
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
echo -e "\e[36mDo you wish to automatically clean up background noise? Please note this function is a little quirky. Use at your own risk. (y/n)\e[0m"
read filter_choice
if [ "$filter_choice" = "y" ]; then
    python3 filter.py
else
    echo "[OK] You can run it manually later with: python3 filter.py"
fi
echo
echo -e "\e[35m=======================================================\e[0m"
echo
echo -e "\e[36mDo you wish to split the audio files into smaller chunks? It's recommended that you do so. (y/n)\e[0m"
read filter_choice
if [ "$filter_choice" = "y" ]; then
    python3 split.py
else
    echo "[OK] You can run it manually later with: python3 split.py"
fi
echo
echo -e "\e[35m=======================================================\e[0m"
echo
echo -e "\e[36mReady to run the generation script to generate the dataset? (y/n)\e[0m"
read run_choice
if [ "$run_choice" = "y" ]; then
    python3 main.py
else
    echo "[OK] You can run it manually later with: python3 main.py"
fi
echo
echo -e "\e[35m=======================================================\e[0m"
echo
#deactivate venv
deactivate

if [ -f "output/dataset.zip" ]; then
    echo -e "[OK] \e[32mDataset Generation Complete! You should now see a zip folder located in output/ named dataset.zip\e[0m"
else
    echo -e "\e[31m=======================================================\e[0m"
    echo -e "[ERROR] \e[31mDataset generation failed! The expected file output/dataset.zip does not exist.\e[0m"
    echo
    echo "[ERROR] \e[31mConsider making an issue thread at https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues !\e[0m"
    echo -e "\e[31m=======================================================\e[0m"
fi

