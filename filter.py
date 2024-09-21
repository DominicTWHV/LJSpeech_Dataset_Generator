import os
import noisereduce as nr
import librosa
import soundfile as sf

#input dir, do NOT edit unless you wish to get a different name for some reason
input_dir = 'input/'

def process_audio_files():
    #find all files that are dirty
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav') and not f.endswith('_cleaned.wav')]
    
    for file in files:
        file_path = os.path.join(input_dir, file)
        
        #load file
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        print(f"[DEBUG]: Loading {file_path} for cleaning...")
        
        #reduce noise
        reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate)
        
        #rename
        new_filename = file.replace('.wav', '_cleaned.wav')
        new_file_path = os.path.join(input_dir, new_filename)
        
        #save
        sf.write(new_file_path, reduced_noise, sample_rate)
        
        #remove original
        os.remove(file_path)
        
    print(f"Processed and cleaned {len(files)} files.")

if __name__ == '__main__':
    process_audio_files()
