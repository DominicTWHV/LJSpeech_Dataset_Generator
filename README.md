# LJSpeech Dataset Generator

Designed for use with PiperTTS (custom model training), uses a standard file structure:
```
dataset.zip/
│
├── metadata.csv
└── wavs/
    ├── <name>-<index>.wav
    └──  ...
```

# Setup:

Script tested on Ubuntu Server 22.04 LTS with python version `3.10.12`

**Update APT Packages:**

```
sudo apt update
sudo apt upgrade
sudo reboot
```

**Clone repository:**

```
git clone https://github.com/DominicTWHV/LJSpeech_Dataset_Generator.git
```

**Prepare an Environment:**

```
sudo apt install python3 python3-venv python3-pip ffmpeg zip
cd LJSpeech_Dataset_Generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Prepare Sample Audio:**

Move all your `.wav` files into input/

**Running:**

```
python3 main.py
```

# Post:

You should now see a zip file (`dataset.zip`) located in the root directory of the project folder, that should contain all files with the standard LJSpeech layout.


