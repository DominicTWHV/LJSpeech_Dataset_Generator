import os
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import concurrent.futures
import threading
import sys
from io import StringIO

class NoiseReducer:
    def __init__(self, input_dir='wavs/'):
        self.input_dir = input_dir
        self.print_lock = threading.Lock()
        # Create an in-memory stream to capture print statements
        self.log_stream = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.log_stream  # Redirect print statements to log_stream

    def apply_dynamic_noise_reduction(self, audio_data, sample_rate, frame_length=2048, hop_length=512):
        energy = np.array([sum(abs(audio_data[i:i+frame_length]**2)) for i in range(0, len(audio_data), hop_length)])
        max_energy = max(energy)
        normalized_energy = energy / max_energy
        silence_threshold = 0.1

        noise_frames = [audio_data[i:i+frame_length] for i in range(0, len(audio_data), hop_length) if normalized_energy[i // hop_length] < silence_threshold]
        
        if noise_frames:
            noise_profile = np.concatenate(noise_frames)
        else:
            noise_profile = audio_data[:frame_length]

        reduced_audio = np.array(audio_data)
        for i in range(0, len(audio_data), hop_length):
            start_idx = i
            end_idx = min(i + frame_length, len(audio_data))
            frame = audio_data[start_idx:end_idx]

            if normalized_energy[i // hop_length] < silence_threshold:
                reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=1.0)
            else:
                reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=0.5)

            reduced_audio[start_idx:end_idx] = reduced_frame

        return reduced_audio

    def process_single_audio_file(self, file):
        file_path = os.path.join(self.input_dir, file)
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        with self.print_lock:
            print(f"[DEBUG] Processing {file_path}.")

        reduced_noise = self.apply_dynamic_noise_reduction(audio_data, sample_rate)
        new_filename = file.replace('.wav', '_cleaned.wav')
        new_file_path = os.path.join(self.input_dir, new_filename)
        sf.write(new_file_path, reduced_noise, sample_rate)
        os.remove(file_path)

    def process_audio_files(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.wav') and not f.endswith('_cleaned.wav')]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_audio_file, file) for file in files]
            concurrent.futures.wait(futures)

        with self.print_lock:
            print(f"[DEBUG] Processed and cleaned {len(files)} files.")

    def get_logs(self):
        # Retrieve the captured logs from the StringIO stream
        logs = self.log_stream.getvalue()
        # Reset the log stream for the next capture
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        return logs
    
    def gradio_run(self):
        self.process_audio_files()
        return self.get_logs()