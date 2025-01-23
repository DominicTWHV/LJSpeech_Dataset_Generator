import os
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
from functions.helper.run_san import check_wav_files

class NoiseReducer:
    def __init__(self, input_dir='wavs/'):
        self.input_dir = input_dir
        self.logs = []  # Initialize the logs list

    def apply_dynamic_noise_reduction(self, audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        self.logs.append(f"[DEBUG] Starting noise reduction with frame_length={frame_length}, hop_length={hop_length}, silence_threshold={silence_threshold}.")
        yield '\n'.join(self.logs)
        
        # Calculate energy of each frame
        energy = np.array([sum(abs(audio_data[i:i+frame_length]**2)) for i in range(0, len(audio_data), hop_length)])
        max_energy = max(energy) if energy.size > 0 else 1  # Avoid division by zero
        normalized_energy = energy / max_energy
        
        # Determine noise frames
        noise_frames = [audio_data[i:i+frame_length] for i in range(0, len(audio_data), hop_length) if (i // hop_length) < len(normalized_energy) and normalized_energy[i // hop_length] < silence_threshold]
        if noise_frames:
            noise_profile = np.concatenate(noise_frames)
        else:
            noise_profile = audio_data[:frame_length]
        
        reduced_audio = np.array(audio_data)
        for i in range(0, len(audio_data), hop_length):
            start_idx = i
            end_idx = min(i + frame_length, len(audio_data))
            frame = audio_data[start_idx:end_idx]
            if (i // hop_length) < len(normalized_energy) and normalized_energy[i // hop_length] < silence_threshold:
                reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=prop_decrease_noisy)
            else:
                reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=prop_decrease_normal)
            reduced_audio[start_idx:end_idx] = reduced_frame
        
        self.logs.append(f"[DEBUG] Completed noise reduction for current audio data.")
        yield '\n'.join(self.logs)
        return reduced_audio

    def process_single_audio_file(self, file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        file_path = os.path.join(self.input_dir, file)
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        self.logs.append(f"[DEBUG] Processing {file_path}. Sample rate: {sample_rate}, Audio length: {len(audio_data)}")
        yield '\n'.join(self.logs)

        reduced_audio_generator = self.apply_dynamic_noise_reduction(
            audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal
        )

        reduced_audio = None
        for result in reduced_audio_generator:
            if isinstance(result, str):
                self.logs.append(result)
                yield '\n'.join(self.logs)
            else:
                reduced_audio = result

        # Save reduced audio to a new file
        new_filename = file.replace('.wav', '_cleaned.wav')
        new_file_path = os.path.join(self.input_dir, new_filename)
        sf.write(new_file_path, reduced_audio, sample_rate)
        os.remove(file_path)
        self.logs.append(f"[DEBUG] Saved cleaned file as {new_file_path} and removed original file {file_path}.")
        yield '\n'.join(self.logs)

    def process_audio_files(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.wav') and not f.endswith('_cleaned.wav')]
        self.logs.append(f"[DEBUG] Found {len(files)} files to process.")
        yield '\n'.join(self.logs)
        
        for file in files:
            try:
                file_generator = self.process_single_audio_file(
                    file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal
                )
                for result in file_generator:
                    self.logs.append(result)
                    yield '\n'.join(self.logs)
            except Exception as e:
                self.logs.append(f"[ERROR] Failed to process {file}: {str(e)}")
                yield '\n'.join(self.logs)

        self.logs.append(f"[DEBUG] Processed and cleaned {len(files)} files.")
        yield '\n'.join(self.logs)

    def gradio_run(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        if not check_wav_files():
            self.logs.append("ERROR: No .wav files found in the input directory. Please upload them and try again.")
            yield '\n'.join(self.logs)
            return
        
        for log in self.process_audio_files(frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
            self.logs.append(log)
            yield '\n'.join(self.logs)
