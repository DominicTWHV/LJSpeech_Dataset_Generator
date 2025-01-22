import sys
from io import StringIO
import os
import re
import pandas as pd
import subprocess
import wave
import speech_recognition as sr

class MainProcess:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Create an in-memory stream to capture print statements
        self.log_stream = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.log_stream  # Redirect print statements to log_stream

    def transcribe_audio(self, audio_file):
        print(f"[DEBUG] Transcribing audio file: {audio_file}")
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                transcript = self.recognizer.recognize_google(audio_data)
                print(f"[DEBUG] Transcript: {transcript}")
            except sr.UnknownValueError:
                print(f"[WARNING] Google Speech Recognition could not understand {audio_file}")
                transcript = ""
            except sr.RequestError as e:
                print(f"[ERROR] Could not request results; {e}\n\nAre you connected to the internet?")
                transcript = ""
        return transcript

    def process_wav_files(self, input_dir):
        print(f"[DEBUG] Processing .wav files in directory: {input_dir}")
        metadata = []
        wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        print(f"[DEBUG] Found {len(wav_files)} .wav files")

        for wav_file in wav_files:
            input_path = os.path.join(input_dir, wav_file)
            print(f"[DEBUG] Processing file: {wav_file}")
            transcript = self.transcribe_audio(input_path)
            if transcript:
                metadata.append([os.path.join('wavs', wav_file), transcript])
            else:
                os.remove(input_path)
                print(f"[WARNING] Skipping file {wav_file} as no transcript is available.")

        metadata.sort(key=lambda x: int(re.search(r'processed(\d+)', x[0]).group(1)))
        print(f"[DEBUG] Writing metadata.csv...")
        df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
        df.to_csv("metadata.csv", sep='|', index=False)

        print(f"[DEBUG] metadata.csv generated successfully")

    def zip_output(self, output_filename="output/dataset.zip"):
        print(f"[DEBUG] Zipping the output files into {output_filename}...")
        try:
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            subprocess.run(['zip', '-r', output_filename, 'wavs', 'metadata.csv'], check=True)
            print(f"[DEBUG] Successfully created {output_filename}")
            return output_filename
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error while zipping: {e}")
            return None

    def total_audio_length(self, directory):
        total_length = 0
        wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        for wav_file in wav_files:
            file_path = os.path.join(directory, wav_file)
            with wave.open(file_path, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
                total_length += duration
        return total_length

    def gradio_run(self):
        self.process_wav_files('wavs')
        return self.get_logs()

    def manual_run(self):
        print(f"[DEBUG] Starting the audio processing pipeline...")
        self.process_wav_files('wavs')
        self.zip_output()
        print(f"[OK] Dataset Created!")
    
    def get_logs(self):
        # Retrieve the captured logs from the StringIO stream
        logs = self.log_stream.getvalue()
        # Reset the log stream for the next capture
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        return logs
