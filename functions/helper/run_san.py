import os

def check_wav_files(directory='wavs/'):
    if not os.path.isdir(directory):
        return False

    for file in os.listdir(directory):
        if file.lower().endswith('.wav'):
            return True
    
    return False
