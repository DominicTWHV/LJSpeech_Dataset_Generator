# Overview

Built for PiperTTS (custom model training) with a simple file structure:

```
dataset.zip/
├── metadata.csv
└── wavs/
    ├── <name>_processed<index>.wav
    └── ...
```

Works with LJSpeech-compatible TTS engines too!  
**metadata.csv** looks like:

```csv
wav_filename|transcript
wavs/<name>_processed<index>.wav|<transcript>
...
```

---

# Setup

Tested on Ubuntu Server 22.04 (Python 3.10.12) and Ubuntu Desktop 24.04 (Python 3.12.3). Should run fine on Debian-based systems.  
**No Windows support, sorry!**

1. Clone it:
   ```
   git clone https://github.com/DominicTWHV/LJSpeech_Dataset_Generator.git
   ```
2. Set it up:
   ```
   cd LJSpeech_Dataset_Generator
   sudo chmod +x pipeline.sh
   ```
3. Run it:
   ```
   ./pipeline.sh
   ```
   Then hop onto the Gradio WebUI @ port 7860. The server listens on 0.0.0.0:7860 by default.

---

# Post-Processing

Move `dataset.zip` to your training directory:
```
mv /output/dataset.zip /path/to/training/dir
unzip dataset.zip
```
Or just download it via the WebUI.

---

# Troubleshooting

File permission issues? Missing files? Check script permissions or background processes.  
Still stuck? Feel free to drop a note in the [issues](https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues) tab.
