import re
import math
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from functions.helper.run_san import check_wav_files


class AudioSplitter:
    def __init__(self, input_dir='wavs/', output_dir='wavs/'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processed_pattern = re.compile(r'^(.*)_processed(\d+)\.wav$')

    def _find_split_points(self, audio, min_chunk_duration, max_chunk_duration):
        # find speech regions
        speech_regions = detect_nonsilent(
            audio,
            min_silence_len=300,
            silence_thresh=audio.dBFS - 16,
            seek_step=10,
        )

        if not speech_regions:
            # fall back to single chunk if no speech
            return [(0, len(audio))]

        total_duration = len(audio)
        chunks = []
        chunk_start = speech_regions[0][0]

        for i, (region_start, region_end) in enumerate(speech_regions):
            current_chunk_len = region_end - chunk_start

            # if adding this region would exceed max_chunk_duration, finalize the current chunk before starting a new one.
            if current_chunk_len > max_chunk_duration and region_start > chunk_start:
                # End the chunk at the silence gap before this region
                chunks.append((chunk_start, region_start))
                chunk_start = region_start

            # if last region, close the final chunk
            if i == len(speech_regions) - 1:
                chunks.append((chunk_start, region_end))

        # merge trailing chunks that are too short into the previous one
        merged = []
        for start, end in chunks:
            if merged and (end - start) < min_chunk_duration:
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, end)
                
            else:
                merged.append((start, end))

        # safety: even duration split if chunk too big
        final = []
        for start, end in merged:
            duration = end - start
            if duration > max_chunk_duration * 1.5:
                n = math.ceil(duration / max_chunk_duration)
                sub_len = math.ceil(duration / n)
                pos = start
                for _ in range(n):
                    sub_end = min(pos + sub_len, end)
                    final.append((pos, sub_end))
                    pos = sub_end

            else:
                final.append((start, end))

        return final

    def split_audio(self, filepath, min_chunk_duration, max_chunk_duration):
        logs = []
        filepath = Path(filepath)
        audio = AudioSegment.from_wav(filepath)
        total_duration = len(audio)
        if total_duration <= 0:
            logs.append(f"[WARNING] Skipping empty file: {filepath.name}")
            return logs, False

        # if file is already within acceptable range, skip splitting
        if total_duration <= max_chunk_duration:
            logs.append(f"[DEBUG] {filepath.name} ({total_duration} ms) is within max duration, skipping split")
            return logs, False

        chunks = self._find_split_points(audio, min_chunk_duration, max_chunk_duration)
        exported_chunks = 0

        logs.append(f"[DEBUG] Splitting {filepath.name} into {len(chunks)} chunks at silence boundaries")

        for i, (start_ms, end_ms) in enumerate(chunks):
            chunk_audio = audio[start_ms:end_ms]
            out_name = f"{filepath.stem}_processed{i+1}.wav"
            output_path = self.output_dir / out_name
            try:
                chunk_audio.export(output_path, format="wav")
                exported_chunks += 1
                duration_ms = end_ms - start_ms
                logs.append(f"[DEBUG] Exported chunk {i+1}: {out_name} ({duration_ms} ms)")

            except Exception as e:
                logs.append(f"[ERROR] Failed to export chunk {i+1} ({out_name}): {e}")
                continue

        return logs, exported_chunks > 0

    def process_directory(self, min_chunk_duration, max_chunk_duration):
        if min_chunk_duration <= 0 or max_chunk_duration <= 0:
            yield "[ERROR] Chunk durations must be greater than zero."
            return

        if min_chunk_duration > max_chunk_duration:
            yield "[ERROR] Minimum chunk duration cannot be greater than maximum chunk duration."
            return

        if not self.input_dir.is_dir():
            yield f"[ERROR] Input directory not found: {self.input_dir}"
            return

        files_to_process = sorted(
            p for p in self.input_dir.iterdir()
            if p.suffix.lower() == '.wav' and not self.processed_pattern.match(p.name)
        )

        file_count = len(files_to_process)
        successfully_split = []

        for filepath in files_to_process:
            yield f"[DEBUG] Splitting {filepath.name}"
            try:
                split_logs, was_split = self.split_audio(filepath, min_chunk_duration, max_chunk_duration)
                for log in split_logs:
                    yield log

                if was_split:
                    successfully_split.append(filepath)

            except Exception as e:
                yield f"[ERROR] Failed to split {filepath.name}: {e}"

        for filepath in successfully_split:
            if filepath.is_file():
                try:
                    filepath.unlink()
                    yield f"[DEBUG] Removed original: {filepath.name}"

                except OSError as e:
                    yield f"[ERROR] Failed to remove {filepath.name}: {e}"
            
        yield f"\n[OK] Finished splitting {file_count} files."

    def gradio_run(self, min_chunk_duration, max_chunk_duration):
        if not check_wav_files():
            yield "ERROR: No .wav files found in the input directory. Please upload them and try again."
            return

        logs = []
        try:
            for log in self.process_directory(min_chunk_duration, max_chunk_duration):
                logs.append(log)
                yield "\n".join(logs)

        except Exception as e:
            yield f"[ERROR] An error occurred during processing: {str(e)}"