# LJSpeech Dataset Generator

Designed for use with PiperTTS (custom model training), using a standard file structure:

```
dataset.zip/
│
├── metadata.csv
└── wavs/
    ├── <name>_processed<index>.wav
    └──  ...
```
_Should also be compatible with other TTS engines that uses a LJSpeech file structure for training._

**metadata.csv:**

```csv
wav_filename|transcript
wavs/<name>_processed<index>.wav|<transcript>
wavs/<name>_processed<index>.wav|<transcript>
...
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
./pipeline.sh
```

Move all your `.wav` files into wavs/ when instructed to do so

# Post:

You should now see a zip file (`dataset.zip`) located in the root directory of the project folder, that should contain all files with the standard LJSpeech layout.


