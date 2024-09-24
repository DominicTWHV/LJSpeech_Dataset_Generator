import os
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import concurrent.futures
import threading

#define the input directory
input_dir = 'wavs/'
#lock to safely print debug messages in multithreaded environment
print_lock = threading.Lock()

def apply_dynamic_noise_reduction(audio_data, sample_rate, frame_length=2048, hop_length=512):
    #calculate short-term energy for each frame
    energy = np.array([
        sum(abs(audio_data[i:i+frame_length]**2))
        for i in range(0, len(audio_data), hop_length)
    ])

    #normalize energy
    max_energy = max(energy)
    normalized_energy = energy / max_energy

    #threshold for detecting silence/background noise (tuneable parameter)
    silence_threshold = 0.1

    #assume that the quieter sections are dominated by noise and calculate the noise profile
    noise_frames = [audio_data[i:i+frame_length] for i in range(0, len(audio_data), hop_length) if normalized_energy[i // hop_length] < silence_threshold]
    
    #if noise frames were found, calculate a noise profile
    if len(noise_frames) > 0:
        noise_profile = np.concatenate(noise_frames)
    else:
        noise_profile = audio_data[:frame_length]  #default to the first frame if no quiet sections are found

    #apply noise reduction
    reduced_audio = np.array(audio_data)
    for i in range(0, len(audio_data), hop_length):
        start_idx = i
        end_idx = min(i + frame_length, len(audio_data))
        frame = audio_data[start_idx:end_idx]

        #determine how much noise reduction to apply based on frame energy
        if normalized_energy[i // hop_length] < silence_threshold:
            #apply more aggressive noise reduction in quieter sections
            reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=1.0)
        else:
            #apply less aggressive reduction in louder sections to avoid cutting off the main signal
            reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=0.5)

        #replace the original frame with the reduced version
        reduced_audio[start_idx:end_idx] = reduced_frame

    return reduced_audio

def process_single_audio_file(file):
    file_path = os.path.join(input_dir, file)

    #load the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    with print_lock:
        print(f"[DEBUG]: Processing {file_path}.")

    #apply dynamic noise reduction
    reduced_noise = apply_dynamic_noise_reduction(audio_data, sample_rate)

    #create new filename with _cleaned suffix
    new_filename = file.replace('.wav', '_cleaned.wav')
    new_file_path = os.path.join(input_dir, new_filename)

    #save the cleaned audio
    sf.write(new_file_path, reduced_noise, sample_rate)

    #remove the original file
    os.remove(file_path)

def process_audio_files():
    #list all .wav files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav') and not f.endswith('_cleaned.wav')]

    #use ThreadPoolExecutor to process files concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        #submit tasks to the thread pool
        futures = [executor.submit(process_single_audio_file, file) for file in files]
        
        #wait for all tasks to complete
        concurrent.futures.wait(futures)

    with print_lock:
        print(f"[DEBUG]: Processed and cleaned {len(files)} files.")

if __name__ == '__main__':
    process_audio_files()
