import os
import re
import math
import tempfile
from pydub import AudioSegment
import speech_recognition as sr
import sys
from io import StringIO

class AudioSplitter:
    def __init__(self, input_dir='wavs/', output_dir='wavs/', max_chunk_duration=8000, min_chunk_duration=4000):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_chunk = max_chunk_duration
        self.min_chunk = min_chunk_duration
        self.processed_pattern = re.compile(r'^(.*)_processed(\d+)\.wav$')
        self.recognizer = sr.Recognizer()
        
        # Create an in-memory stream to capture print statements
        self.log_stream = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.log_stream  # Redirect print statements to log_stream

    def split_audio(self, filepath):
        audio = AudioSegment.from_wav(filepath)
        total_duration = len(audio)
        full_chunks = total_duration // self.max_chunk
        leftover = total_duration % self.max_chunk

        if leftover < self.min_chunk and full_chunks > 0:
            num_chunks = full_chunks
        else:
            num_chunks = full_chunks + 1 if leftover else full_chunks

        chunk_length = math.ceil(total_duration / num_chunks)
        current_pos = 0

        for i in range(num_chunks):
            end_pos = min(current_pos + chunk_length, total_duration)
            chunk_audio = audio[current_pos:end_pos]
            self._maybe_recognize(chunk_audio)
            out_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_processed{i+1}.wav"
            chunk_audio.export(os.path.join(self.output_dir, out_name), format="wav")
            current_pos = end_pos
            if current_pos >= total_duration:
                break

    def _maybe_recognize(self, chunk_audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tfile:
            chunk_audio.export(tfile.name, format="wav")
            with sr.AudioFile(tfile.name) as source:
                audio_data = self.recognizer.record(source)
                try:
                    self.recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    pass

    def process_directory(self):
        file_count = len([f for f in os.listdir(self.input_dir) if f.endswith('.wav') and not self.processed_pattern.match(f)])
        print(f"[DEBUG] Starting to process: {file_count} files.")

        for filename in os.listdir(self.input_dir):
            filepath = os.path.join(self.input_dir, filename)
            if filename.endswith('.wav') and not self.processed_pattern.match(filename):
                print(f"[DEBUG] Splitting {filename}")
                self.split_audio(filepath)

        for filename in os.listdir(self.input_dir):
            filepath = os.path.join(self.input_dir, filename)
            if os.path.isfile(filepath) and not self.processed_pattern.match(filename):
                print(f"[DEBUG] Removing unprocessed file: {filename}")
                os.remove(filepath)

    def get_logs(self):
        # Retrieve the captured logs from the StringIO stream
        logs = self.log_stream.getvalue()
        # Reset the log stream for the next capture
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        return logs
    
    def gradio_run(self):
        self.process_directory()
        print("============================END OF SPLITTING============================")
        return self.get_logs()