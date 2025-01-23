import os
import re
import math
import tempfile
from pydub import AudioSegment
import speech_recognition as sr

from functions.helper.run_san import check_wav_files

class AudioSplitter:
    def __init__(self, input_dir='wavs/', output_dir='wavs/'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processed_pattern = re.compile(r'^(.*)_processed(\d+)\.wav$')
        self.recognizer = sr.Recognizer()

    def split_audio(self, filepath, min_chunk_duration, max_chunk_duration):
        audio = AudioSegment.from_wav(filepath)
        total_duration = len(audio)
        full_chunks = total_duration // max_chunk_duration
        leftover = total_duration % max_chunk_duration

        if leftover < min_chunk_duration and full_chunks > 0:
            num_chunks = full_chunks
        else:
            num_chunks = full_chunks + 1 if leftover else full_chunks

        chunk_length = math.ceil(total_duration / num_chunks)
        current_pos = 0

        yield f"[DEBUG] Splitting {os.path.basename(filepath)} into {num_chunks} chunks, each approximately {chunk_length} ms long."

        for i in range(num_chunks):
            end_pos = min(current_pos + chunk_length, total_duration)
            chunk_audio = audio[current_pos:end_pos]

            try:
                self._maybe_recognize(chunk_audio)
            except Exception as e:
                yield f"[ERROR] Error during speech recognition in chunk {i+1}: {str(e)}"

            out_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_processed{i+1}.wav"
            output_path = os.path.join(self.output_dir, out_name)
            try:
                chunk_audio.export(output_path, format="wav")
                yield f"[DEBUG] Exported chunk {i+1} as {out_name}"
            except Exception as e:
                yield f"[ERROR] Failed to export chunk {i+1} as {out_name}: {str(e)}"
                continue  # Skip to the next chunk if export fails

            current_pos = end_pos
            if current_pos >= total_duration:
                break

    def _maybe_recognize(self, chunk_audio):
        # Use a temporary file to store the chunk for recognition
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tfile:
            chunk_audio.export(tfile.name, format="wav")
            with sr.AudioFile(tfile.name) as source:
                audio_data = self.recognizer.record(source)
                try:
                    self.recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    pass

    def process_directory(self, min_chunk_duration, max_chunk_duration):
        files_to_process = [
            f for f in os.listdir(self.input_dir)
            if f.endswith('.wav') and not self.processed_pattern.match(f)
        ]
        file_count = len(files_to_process)

        # Process each file
        for filename in files_to_process:
            filepath = os.path.join(self.input_dir, filename)
            yield f"[DEBUG] Splitting {filename}"
            try:
                for log in self.split_audio(filepath, min_chunk_duration, max_chunk_duration):
                    yield log
            except Exception as e:
                yield f"[ERROR] Failed to split {filename}: {str(e)}"

        for filename in files_to_process:
            filepath = os.path.join(self.input_dir, filename)
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    yield f"[DEBUG] Removed unprocessed file: {filename}"
                except Exception as e:
                    yield f"[ERROR] Failed to remove unprocessed file {filename}: {str(e)}"
            
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