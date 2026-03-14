from pathlib import Path


def check_wav_files(directory='wavs/'):
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return False
    return any(f.suffix.lower() == '.wav' for f in dir_path.iterdir() if f.is_file())
