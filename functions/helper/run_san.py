import os

def check_wav_files(directory='wavs/'):
    if not os.path.exists(directory):
        return False

    for file in os.listdir(directory):
        if file.endswith('.wav'):
            return True
    
    return False
