import os
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import concurrent.futures
import threading
import sys
from io import StringIO

from functions.helper.run_san import check_wav_files

class NoiseReducer:
    def __init__(self, input_dir='wavs/'):
        self.input_dir = input_dir
        self.print_lock = threading.Lock()

    def apply_dynamic_noise_reduction(self, audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        print(f"[DEBUG] Starting noise reduction with frame_length={frame_length}, hop_length={hop_length}, silence_threshold={silence_threshold}.")
        
        # Calculate energy of each frame
        energy = np.array([sum(abs(audio_data[i:i+frame_length]**2)) for i in range(0, len(audio_data), hop_length)])
        max_energy = max(energy)
        normalized_energy = energy / max_energy

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
                reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=prop_decrease_noisy)
            else:
                reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=prop_decrease_normal)

            reduced_audio[start_idx:end_idx] = reduced_frame

        print(f"[DEBUG] Completed noise reduction for current audio data.")
        
        return reduced_audio

    def process_single_audio_file(self, file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        file_path = os.path.join(self.input_dir, file)
        audio_data, sample_rate = librosa.load(file_path, sr=None)

        print(f"[DEBUG] Processing {file_path}. Sample rate: {sample_rate}, Audio length: {len(audio_data)}")

        reduced_noise = self.apply_dynamic_noise_reduction(audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal)
        new_filename = file.replace('.wav', '_cleaned.wav')
        new_file_path = os.path.join(self.input_dir, new_filename)
        sf.write(new_file_path, reduced_noise, sample_rate)
        os.remove(file_path)

        print(f"[DEBUG] Saved cleaned file as {new_file_path} and removed original file {file_path}.")

    def process_audio_files(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.wav') and not f.endswith('_cleaned.wav')]
        print(f"[DEBUG] Found {len(files)} files to process.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_audio_file, file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal) for file in files]
            concurrent.futures.wait(futures)

        print(f"[DEBUG] Processed and cleaned {len(files)} files.")

    def gradio_run(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        if not check_wav_files():
            return "ERROR: No .wav files found in the input directory. Please upload them and try again."
        
        log_stream = StringIO()
        original_stdout = sys.stdout
        sys.stdout = log_stream

        try:
            self.process_audio_files(frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal)
        finally:
            #ensure stdout restored
            sys.stdout = original_stdout

        logs = log_stream.getvalue()
        log_stream.close()
        return logs