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
You may also modify the `sep='|'` argument in line 56 (main.py) to whatever seperator your TTS engine uses.

You may also modify the chunk length in split.py if your TTS engine requires a specific chunk duration.

_PiperTTS uses_ `|`, _which is the default given here_

-----------------------------------

# Setup:

Script tested on:

Ubuntu Server 22.04 LTS with python version `3.10.12`
Ubuntu Desktop 24.04 LTS with python version `3.12.3`

Should also be fully compatible with Debian based systems.

**Clone repository:**

```
git clone https://github.com/DominicTWHV/LJSpeech_Dataset_Generator.git
```

**Setting up:**

```
cd LJSpeech_Dataset_Generator
sudo chmod +x pipeline.sh
```

**Running:**

```
./pipeline.sh
```

Move all your `.wav` files into wavs/ when instructed to do so, and then follow the (y/n) prompts.

Note: the sanity check script will be ran automatically at the end with no user prompting required.

-----------------------------------

# Post:

You should now see a zip file (`dataset.zip`) located in `output/`, that should contain all files with the standard LJSpeech layout.

You are free to move `dataset.zip` into the input directory of piper (or other training platform) for pre-processing.

```sh
mv /output/dataset.zip /path/to/training/dir/for/pre-processing
```

And unzip it

```sh
unzip dataset.zip
```

# Error reporting:

If sanity check fails, please make sure your audio is clear, as that could indicate one or more chunks does NOT contain any transcript. You should also be able to tell which one it is by looking at the print lines during transcription. (Depending on which error it gives, the error you are looking for is `Critical Error: Mismatch! Expected <num>, but found <num> .wav files.`)

For other errors (less common), check if the script has permission to generate files, or if you/a running background script had removed the generated file(s).

If you require further help, please feel free to make a report under the [issues](https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues) tab.
