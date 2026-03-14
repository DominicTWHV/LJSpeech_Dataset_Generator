import os
import re
import pandas as pd
import wave
import zipfile
from pathlib import Path

from functions.helper.run_san import check_wav_files


class ASREngine:
    ENGINE_LOCAL = "Local (faster-whisper)"
    ENGINE_GOOGLE = "Google Speech API"
    AVAILABLE_ENGINES = [ENGINE_LOCAL, ENGINE_GOOGLE]

    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
    DEVICES = ["auto", "cpu", "cuda"]
    LANGUAGES = [
        "auto", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he",
        "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
        "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv",
        "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy",
        "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
        "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
        "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb",
        "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
        "su", "yue",
    ]

    def __init__(self):
        self.engine = self.ENGINE_LOCAL
        self.model_size = "base"
        self.language = None  # None → auto-detect
        self.device = "auto"
        self.compute_type = "auto"
        self._model = None
        self._batched_pipeline = None

    def configure(self, engine, model_size="base", language="auto", device="auto"):
        needs_reload = (
            self.engine != engine
            or self.model_size != model_size
            or self.device != device
        )
        if needs_reload:
            self._model = None
            self._batched_pipeline = None

        self.engine = engine
        self.model_size = model_size
        self.language = language if language and language != "auto" else None
        self.device = device

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._batched_pipeline = BatchedInferencePipeline(model=self._model)

    def transcribe(self, audio_file):
        if self.engine == self.ENGINE_LOCAL:
            return self._transcribe_local(audio_file)
        return self._transcribe_google(audio_file)

    def _transcribe_local(self, audio_file):
        logs = []
        detected_language = None
        name = Path(audio_file).name
        logs.append(f"[DEBUG] Transcribing ({self.model_size}): {name}")
        try:
            self._ensure_model()
            segments, info = self._batched_pipeline.transcribe(
                str(audio_file),
                batch_size=16,
                language=self.language,
            )
            detected_language = info.language
            logs.append(
                f"[DEBUG] Language: {detected_language} ({info.language_probability:.0%})"
            )
            transcript = " ".join(seg.text for seg in segments).strip()
            if transcript:
                logs.append(f"[DEBUG] Transcript: {transcript}")
            else:
                transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
                logs.append(f"[WARNING] Empty transcript for {name}")
        except Exception as e:
            logs.append(f"[ERROR] Transcription failed for {name}: {e}")
            transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
        return transcript, detected_language, logs

    def _transcribe_google(self, audio_file):
        import speech_recognition as sr
        logs = []
        name = Path(audio_file).name
        logs.append(f"[DEBUG] Transcribing (Google API): {name}")
        recognizer = sr.Recognizer()
        detected_language = None
        try:
            with sr.AudioFile(str(audio_file)) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
                logs.append(f"[DEBUG] Transcript: {transcript}")
        except sr.UnknownValueError:
            logs.append(f"[WARNING] Could not understand audio: {name}")
            transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
        except sr.RequestError as e:
            logs.append(f"[ERROR] API request failed: {e}\n\nAre you connected to the internet?")
            transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
        except Exception as e:
            logs.append(f"[ERROR] Transcription failed: {e}")
            transcript = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"
        return transcript, detected_language, logs


class MainProcess:
    def __init__(self, input_dir='wavs', metadata_file='metadata.csv', output_dir='output'):
        self.input_dir = Path(input_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.asr = ASREngine()

    @staticmethod
    def _sort_key(wav_path):
        match = re.search(r'processed(\d+)', wav_path.stem)
        if match:
            prefix = wav_path.stem[:match.start()].rstrip('_-').lower()
            return (prefix, 0, int(match.group(1)), wav_path.name.lower())
        return (wav_path.stem.lower(), 1, 0, wav_path.name.lower())

    def _get_wav_files(self, directory):
        directory = Path(directory)
        return sorted(
            [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == '.wav'],
            key=self._sort_key,
        )

    def process_wav_files(self, input_dir=None, separator='|'):
        directory = Path(input_dir) if input_dir else self.input_dir
        if not directory.exists():
            yield f"[ERROR] Input directory not found: {directory}"
            return

        wav_files = self._get_wav_files(directory)
        yield f"[DEBUG] Found {len(wav_files)} .wav files. Using '{separator}' as separator."
        yield f"[DEBUG] ASR: {self.asr.engine} | Model: {self.asr.model_size} | Language: {self.asr.language or 'auto-detect'}"

        metadata = []
        languages_detected = set()

        for wav_file in wav_files:
            yield f"[DEBUG] Processing: {wav_file.name}"
            transcript, detected_lang, transcribe_logs = self.asr.transcribe(wav_file)
            for log in transcribe_logs:
                yield log
            if detected_lang:
                languages_detected.add(detected_lang)
            metadata.append([Path('wavs', wav_file.name).as_posix(), transcript])

        yield "[DEBUG] Writing metadata.csv..."
        df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
        df.to_csv(self.metadata_file, sep=separator, index=False, header=False)

        lang_summary = ", ".join(sorted(languages_detected)) if languages_detected else "N/A"
        yield f"[DEBUG] Languages detected: {lang_summary}"
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
        directory = Path(directory)
        for wav_file in directory.iterdir():
            if wav_file.suffix.lower() == '.wav':
                try:
                    with wave.open(str(wav_file), 'rb') as wf:
                        total_length += wf.getnframes() / wf.getframerate()
                except Exception:
                    continue
        return total_length

    def gradio_run(self, separator, asr_engine, model_size, language, device):
        if not check_wav_files():
            yield "ERROR: No .wav files found in the input directory. Please upload them and try again."
            return

        self.asr.configure(engine=asr_engine, model_size=model_size, language=language, device=device)

        logs = []
        for log in self.process_wav_files(separator=separator):
            logs.append(log)
            yield "\n".join(logs)