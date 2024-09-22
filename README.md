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

**Clone repository:**

```
git clone https://github.com/DominicTWHV/LJSpeech_Dataset_Generator.git
```

**Prepare an Environment:**

```
cd LJSpeech_Dataset_Generator
chmod +x pipeline.sh
```

**Run pipeline:**

```
sudo ./pipeline.sh
```

Move all your `.wav` files into input/ when instructed to do so

# Post:

You should now see a zip file (`dataset.zip`) located in the root directory of the project folder, that should contain all files with the standard LJSpeech layout.


