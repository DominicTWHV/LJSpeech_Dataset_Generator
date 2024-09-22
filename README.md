# Overview

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
You may also modify the `sep='|'` argument in line 52 to whatever seperator your TTS engine uses.

_PiperTTS uses `|`, which is the default given here_

-----------------------------------

# Setup:

Script tested on Ubuntu Server 22.04 LTS with python version `3.10.12`

**Clone repository:**

```
git clone https://github.com/DominicTWHV/LJSpeech_Dataset_Generator.git
```

**Setting up:**

```
cd LJSpeech_Dataset_Generator
chmod +x pipeline.sh
```

**Running:**

```
./pipeline.sh
```

Move all your `.wav` files into wavs/ when instructed to do so, and then follow the (y/n) prompts.

-----------------------------------

# Post:

You should now see a zip file (`dataset.zip`) located in the root directory of the project folder, that should contain all files with the standard LJSpeech layout.


