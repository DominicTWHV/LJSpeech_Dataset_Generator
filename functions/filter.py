import os
import tempfile
import noisereduce as nr
import soundfile as sf
import numpy as np
from pathlib import Path

from functions.helper.run_san import check_wav_files


class NoiseReducer:
    def __init__(self, input_dir='wavs/'):
        self.input_dir = Path(input_dir)

    @staticmethod
    def _estimate_noise_profile(audio, sample_rate, frame_length, hop_length, silence_threshold):
        """Estimate a noise profile from quiet sections of the audio."""
        rms_values = []
        indices = list(range(0, len(audio), hop_length))

        for i in indices:
            frame = audio[i:i + frame_length]
            if len(frame) > 0:
                rms_values.append(np.sqrt(np.mean(frame ** 2)))
            else:
                rms_values.append(0.0)

        rms_values = np.array(rms_values)
        max_rms = np.max(rms_values) if rms_values.size > 0 else 0

        if max_rms <= 0:
            return audio[:max(frame_length, len(audio) // 10)]

        normalized_rms = rms_values / max_rms
        noise_frames = []
        for idx, i in enumerate(indices):
            if idx < len(normalized_rms) and normalized_rms[idx] < silence_threshold:
                noise_frames.append(audio[i:i + frame_length])

        if noise_frames:
            return np.concatenate(noise_frames)
        return audio[:max(frame_length, len(audio) // 10)]

    def reduce_noise_single_pass(self, audio_data, sample_rate, frame_length,
                                  hop_length, silence_threshold, noise_reduction_strength):
        """Single-pass noise reduction that filters noise while retaining crisp speech."""
        logs = []
        if len(audio_data) == 0:
            logs.append("[WARNING] Audio file is empty; skipping noise reduction")
            return audio_data, logs

        original_dtype = audio_data.dtype
        audio_float = audio_data.astype(np.float32)
        if np.issubdtype(original_dtype, np.integer):
            audio_float = audio_float / np.iinfo(original_dtype).max

        noise_profile = self._estimate_noise_profile(
            audio_float, sample_rate, frame_length, hop_length, silence_threshold
        )
        logs.append(f"[DEBUG] Noise profile: {len(noise_profile)} samples from quiet sections")

        reduced = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            y_noise=noise_profile,
            prop_decrease=noise_reduction_strength,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        logs.append("[DEBUG] Single-pass noise reduction complete")

        if np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
            reduced = np.clip(reduced * max_val, np.iinfo(original_dtype).min, max_val)

        return reduced.astype(original_dtype), logs

    def process_single_audio_file(self, file_path, frame_length, hop_length,
                                   silence_threshold, noise_reduction_strength,
                                   use_spectral_gating):
        """Process a single audio file with atomic temp-file writes."""
        file_path = Path(file_path)

        try:
            audio_data, sample_rate = sf.read(file_path)
        except Exception as e:
            yield f"[ERROR] Failed to read {file_path.name}: {e}"
            return

        if audio_data.ndim > 1:
            yield f"[DEBUG] Converting {audio_data.shape[1]}-channel audio to mono"
            audio_data = np.mean(audio_data, axis=1)

        yield f"[DEBUG] Processing: {file_path.name} (sr={sample_rate})"

        try:
            if use_spectral_gating:
                yield "[DEBUG] Using spectral gating for noise reduction"
                reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
            else:
                reduced_audio, reduction_logs = self.reduce_noise_single_pass(
                    audio_data, sample_rate, frame_length, hop_length,
                    silence_threshold, noise_reduction_strength,
                )
                for log in reduction_logs:
                    yield log
        except Exception as e:
            yield f"[ERROR] Noise reduction failed for {file_path.name}: {e}"
            return

        # Write to temp file, then atomically replace original
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav', dir=file_path.parent)
            os.close(tmp_fd)
            sf.write(tmp_path, reduced_audio, sample_rate)
            os.replace(tmp_path, file_path)
            yield f"[DEBUG] Cleaned: {file_path.name}"
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            yield f"[ERROR] Failed to save {file_path.name}: {e}"
            return

        yield "=====================================\n"

    def process_audio_files(self, frame_length, hop_length, silence_threshold,
                            noise_reduction_strength, use_spectral_gating):
        """Process all WAV files in the input directory."""
        if not self.input_dir.is_dir():
            yield f"[ERROR] Input directory not found: {self.input_dir}"
            return

        files = sorted(
            p for p in self.input_dir.iterdir()
            if p.suffix.lower() == '.wav'
        )
        yield f"[DEBUG] Found {len(files)} files to process.\n"

        for file_path in files:
            try:
                for log in self.process_single_audio_file(
                    file_path, frame_length, hop_length, silence_threshold,
                    noise_reduction_strength, use_spectral_gating,
                ):
                    yield log
            except Exception as e:
                yield f"[ERROR] Failed to process {file_path.name}: {e}"

        yield f"\n[OK] Finished filtering {len(files)} files."

    def gradio_run(self, frame_length, hop_length, silence_threshold,
                   noise_reduction_strength, use_spectral_gating):
        if not check_wav_files():
            yield "ERROR: No .wav files found in the input directory. Please upload them and try again."
            return

        logs = []
        for log in self.process_audio_files(
            frame_length, hop_length, silence_threshold,
            noise_reduction_strength, use_spectral_gating,
        ):
            logs.append(log)
            yield "\n".join(logs)