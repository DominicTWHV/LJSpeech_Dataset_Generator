import os
import noisereduce as nr
import soundfile as sf
import numpy as np

from functions.helper.run_san import check_wav_files

class NoiseReducer:
    def __init__(self, input_dir='wavs/'):
        self.input_dir = input_dir

    def apply_dynamic_noise_reduction(self, audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
        logs = []
        
        original_dtype = audio_data.dtype
        audio_float = audio_data.astype(np.float32)
        if np.issubdtype(original_dtype, np.integer):
            audio_float = audio_float / np.iinfo(original_dtype).max
        
        indices = list(range(0, len(audio_float), hop_length))
        energy = []
        
        for i in indices:
            frame_end = min(i + frame_length, len(audio_float))
            frame = audio_float[i:frame_end]
            if len(frame) > 0:
                frame_energy = np.sum(frame**2)
                energy.append(frame_energy)
            else:
                energy.append(0)
        
        energy = np.array(energy)
        max_energy = np.max(energy) if energy.size > 0 else 1
        normalized_energy = energy / max_energy

        noise_frames = []
        for idx, i in enumerate(indices):
            if idx < len(normalized_energy) and normalized_energy[idx] < silence_threshold:
                frame_end = min(i + frame_length, len(audio_float))
                noise_frames.append(audio_float[i:frame_end])
        
        if noise_frames:
            noise_profile = np.concatenate(noise_frames)
        else:
            fallback_length = max(frame_length, len(audio_float) // 10)
            noise_profile = audio_float[:fallback_length]
            logs.append("[WARNING] No quiet frames found, using beginning of audio as noise profile")
        
        reduced_audio = np.array(audio_float)
        
        for idx, i in enumerate(indices):
            if idx >= len(normalized_energy):
                break
                
            start_idx = i
            end_idx = min(i + frame_length, len(audio_float))
            frame = audio_float[start_idx:end_idx]
            
            prop_decrease = prop_decrease_noisy if normalized_energy[idx] < silence_threshold else prop_decrease_normal
            
            try:
                reduced_frame = nr.reduce_noise(y=frame, sr=sample_rate, y_noise=noise_profile, prop_decrease=prop_decrease)
                
                if len(reduced_frame) != len(frame):
                    reduced_frame = np.resize(reduced_frame, frame.shape)
                
                reduced_audio[start_idx:end_idx] = reduced_frame
                
            except Exception as e:
                logs.append(f"[WARNING] Failed to reduce noise for frame {idx}: {str(e)}")
        
        logs.append(f"[DEBUG] Completed noise reduction for current audio data.")
        
        if np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
            reduced_audio = np.clip(reduced_audio * max_val, np.iinfo(original_dtype).min, max_val)
        
        reduced_audio = reduced_audio.astype(original_dtype)
        return reduced_audio, logs

    def process_single_audio_file(self, file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_spectral_gating):
        file_path = os.path.join(self.input_dir, file)
        new_filename = file.replace('.wav', '_cleaned.wav')
        new_file_path = os.path.join(self.input_dir, new_filename)

        try:
            audio_data, sample_rate = sf.read(file_path)
            
            if audio_data.ndim > 1:
                yield f"[DEBUG] Converting {audio_data.shape[1]}-channel audio to mono"
                audio_data = np.mean(audio_data, axis=1)
            
            yield f"[DEBUG] Processing file: {file_path} with sample rate: {sample_rate}"

            if use_spectral_gating:
                yield f"[DEBUG] Using PyTorch Spectral Gating for noise reduction. Processing file: {file_path}"
                reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
            else:
                reduced_audio, reduction_logs = self.apply_dynamic_noise_reduction(
                    audio_data, sample_rate, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal)
                for log in reduction_logs:
                    yield log

            sf.write(new_file_path, reduced_audio, sample_rate)
            os.remove(file_path)
            yield f"[DEBUG] Saved cleaned file as {new_file_path} and removed original file {file_path}."
            
        except Exception as e:
            yield f"[ERROR] Failed to process {file_path}: {str(e)}"
            return
            
        yield "=====================================\n"

    def process_audio_files(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_spectral_gating):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.wav') and not f.endswith('_cleaned.wav')]
        yield f"[DEBUG] Found {len(files)} files to process.\n"

        # Process files sequentially to ensure proper yielding to Gradio
        for file in files:
            try:
                for log in self.process_single_audio_file(file, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_spectral_gating):
                    yield log
            except Exception as e:
                yield f"[ERROR] Failed to process {file}: {str(e)}"

        yield f"\n[OK] Finished filtering {len(files)} files."

    def gradio_run(self, frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_spectral_gating):
        if not check_wav_files():
            yield "ERROR: No .wav files found in the input directory. Please upload them and try again."
            return

        logs = []
        for log in self.process_audio_files(frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_spectral_gating):
            logs.append(log)
            yield "\n".join(logs)