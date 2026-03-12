import os
import re
import pandas as pd
import wave
import zipfile
from pathlib import Path
import speech_recognition as sr

from functions.helper.run_san import check_wav_files

class MainProcess:
    def __init__(self, input_dir='wavs', metadata_file='metadata.csv', output_dir='output'):
        self.recognizer = sr.Recognizer()
        self.input_dir = Path(input_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)

    @staticmethod
    def _sort_key(wav_path):
        match = re.search(r'processed(\d+)', wav_path.stem)
        if match:
            prefix = wav_path.stem[:match.start()].rstrip('_-').lower()
            return (prefix, 0, int(match.group(1)), wav_path.name.lower())
        return (wav_path.stem.lower(), 1, 0, wav_path.name.lower())

    def _get_wav_files(self, directory):
        return sorted(
            [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == '.wav'],
            key=self._sort_key,
        )

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
                transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
            except sr.RequestError as e:
                logs.append(f"[ERROR] Could not request results; {e}\n\nAre you connected to the internet?")
                transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
        return transcript, logs

    def process_wav_files(self, input_dir=None, separator='|'):
        directory = Path(input_dir) if input_dir else self.input_dir
        if not directory.exists():
            yield f"[ERROR] Input directory not found: {directory}"
            return

        wav_files = self._get_wav_files(directory)
        yield f"[DEBUG] Found {len(wav_files)} .wav files. Using '{separator}' as separator for metadata.csv"

        metadata = []

        for wav_file in wav_files:
            input_path = directory / wav_file.name
            yield f"[DEBUG] Processing file: {wav_file.name}"
            transcript, transcribe_logs = self.transcribe_audio(input_path)
            for log in transcribe_logs:
                yield log
            if not transcript:
                transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
                yield f"[WARNING] Empty transcript for {wav_file.name}; using placeholder text."
            metadata.append([Path('wavs', wav_file.name).as_posix(), transcript])

        yield f"[DEBUG] Writing metadata.csv..."
        df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
        df.to_csv(self.metadata_file, sep=separator, index=False, header=False)
        yield f"[DEBUG] metadata.csv generated successfully\n\n[OK] Finished processing {len(metadata)} files."

    def zip_output(self, output_filename=None):
        output_path = Path(output_filename) if output_filename else self.output_dir / 'dataset.zip'
        if not self.metadata_file.is_file() or not self.input_dir.is_dir():
            return None

        wav_files = self._get_wav_files(self.input_dir)
        if not wav_files:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.metadata_file, arcname=self.metadata_file.name)
            for wav_file in wav_files:
                archive.write(wav_file, arcname=Path('wavs', wav_file.name).as_posix())

        return str(output_path)

    def total_audio_length(self, directory):
        total_length = 0
        wav_files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')]
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
        for log in self.process_wav_files(separator=separator):
            logs.append(log)
            yield "\n".join(logs)