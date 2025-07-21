import os
import noisereduce as nr
import soundfile as sf
import numpy as np

from scipy.io import wavfile

from functions.helper.run_san import check_wav_files

class NoiseReducer:
    def __init__(self, input_dir='wavs/'):
        self.input_dir = input_dir

    def apply_dynamic_noise_reduction(self, audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        # Collect logs to yield later
        logs = []

        energy = np.array([np.sum(np.abs(audio_data[i:i+frame_length]**2)) for i in range(0, len(audio_data), hop_length)])
        max_energy = np.max(energy) if energy.size > 0 else 1  # Avoid division by zero
        normalized_energy = energy / max_energy

        indices = [i for i in range(0, len(audio_data), hop_length)]
        noise_frames = [audio_data[i:i+frame_length] for idx, i in enumerate(indices) if normalized_energy[idx] < silence_threshold]

        if noise_frames:
            noise_profile = np.concatenate(noise_frames)
        else:
            noise_profile = audio_data[:frame_length]

        reduced_audio = np.array(audio_data)

        #loop ver audio data
        for idx, i in enumerate(indices):
            start_idx = i
            end_idx = min(i + frame_length, len(audio_data))
            frame = audio_data[start_idx:end_idx]

            if idx < len(normalized_energy):
                if normalized_energy[idx] < silence_threshold:
                    reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=prop_decrease_noisy)
                else:
                    reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=prop_decrease_normal)

                # Ensure that reduced_frame is the same length as frame
                if len(reduced_frame) != len(frame):
                    # Adjust reduced_frame to match the original frame's length
                    reduced_frame = np.resize(reduced_frame, frame.shape)

                reduced_audio[start_idx:end_idx] = reduced_frame
            else:
                logs.append(f"[ERROR] Index {idx} out of bounds for normalized_energy, skipping frame.")

        logs.append(f"[DEBUG] Completed noise reduction for current audio data.")

        #return audio and logs
        return reduced_audio, logs

    def process_single_audio_file(self, file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_hardware_acceleration):
        file_path = os.path.join(self.input_dir, file)
        new_filename = file.replace('.wav', '_cleaned.wav')
        new_file_path = os.path.join(self.input_dir, new_filename)

        yield f"[DEBUG] Processing {file_path}."

        sample_rate, audio_data = wavfile.read(file_path)
        yield f"Sample rate: {sample_rate}, Audio length: {len(audio_data)}"
        
        if use_hardware_acceleration:
            reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, nonstationary=True)
            wavfile.write(new_file_path, sample_rate, reduced_audio)

        else:
            reduced_audio, reduction_logs = self.apply_dynamic_noise_reduction(
                audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal)

            yield from reduction_logs
            wavfile.write(new_file_path, sample_rate, reduced_audio)
            os.remove(file_path)

        yield f"[DEBUG] Saved cleaned file as {new_file_path} and removed original file {file_path}.\n=====================================\n"

    def process_audio_files(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_hardware_acceleration):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.wav') and not f.endswith('_cleaned.wav')]
        yield f"[DEBUG] Found {len(files)} files to process.\n"

        # Process files sequentially to ensure proper yielding to Gradio
        for file in files:
            try:
                for log in self.process_single_audio_file(file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_hardware_acceleration=False):
                    yield log
            except Exception as e:
                yield f"[ERROR] Failed to process {file}: {str(e)}"

        yield f"\n[OK] Finished filtering {len(files)} files."

    def gradio_run(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_hardware_acceleration):
        if not check_wav_files():
            yield "ERROR: No .wav files found in the input directory. Please upload them and try again."
            return

        logs = []
        for log in self.process_audio_files(frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_hardware_acceleration):
            logs.append(log)
            yield "\n".join(logs)