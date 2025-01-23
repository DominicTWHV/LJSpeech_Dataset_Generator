import os
import re
import pandas as pd
import subprocess
import wave
import speech_recognition as sr

from functions.helper.run_san import check_wav_files

class MainProcess:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_file):
        logs = []
        logs.append(f"[DEBUG] Transcribing audio file: {audio_file}")
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                transcript = self.recognizer.recognize_google(audio_data)
                logs.append(f"[DEBUG] Transcript: {transcript}")
            except sr.UnknownValueError:
                logs.append(f"[WARNING] Google Speech Recognition could not understand {audio_file}")
                transcript = ""
            except sr.RequestError as e:
                logs.append(f"[ERROR] Could not request results; {e}\n\nAre you connected to the internet?")
                transcript = ""
        return transcript, logs

    def process_wav_files(self, input_dir, separator=','):
        wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        yield f"[DEBUG] Found {len(wav_files)} .wav files. Using '{separator}' as separator for metadata.csv"

        metadata = []

        for wav_file in wav_files:
            input_path = os.path.join(input_dir, wav_file)
            yield f"[DEBUG] Processing file: {wav_file}"
            transcript, transcribe_logs = self.transcribe_audio(input_path)
            for log in transcribe_logs:
                yield log
            if transcript:
                metadata.append([os.path.join('wavs', wav_file), transcript])
            else:
                os.remove(input_path)
                yield f"[WARNING] Skipping file {wav_file} as no transcript is available."

        metadata.sort(key=lambda x: int(re.search(r'processed(\d+)', x[0]).group(1)))
        yield f"[DEBUG] Writing metadata.csv..."
        df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
        df.to_csv("metadata.csv", sep=separator, index=False)
        yield f"[DEBUG] metadata.csv generated successfully\n\n[OK] Finished processing {len(wav_files)} files."

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

    def gradio_run(self, separator):
        if not check_wav_files():
            yield "ERROR: No .wav files found in the input directory. Please upload them and try again."
            return

        logs = []
        for log in self.process_wav_files('wavs', separator):
            logs.append(log)
            yield "\n".join(logs)