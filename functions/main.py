import os
import re
import ctypes
import site
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import wave
import zipfile
from pathlib import Path

from functions.helper.run_san import check_wav_files


def _load_pip_cuda_libraries():
    lib_dirs = []

    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        for subdir in ("cuda_runtime", "cublas", "cudnn"):
            candidate = base / subdir / "lib"
            if candidate.is_dir() and candidate not in lib_dirs:
                lib_dirs.append(candidate)

    # keep paths visible for debugging needs
    current = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    merged = [str(path) for path in lib_dirs if str(path) not in current] + current
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    # preload by absolute path so dependency resolution works even when loader does not re-read LD_LIBRARY_PATH changes made after process startup.
    preload_names = [
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
    ]

    for lib_name in preload_names:
        for lib_dir in lib_dirs:
            lib_path = lib_dir / lib_name
            if lib_path.is_file():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

                except OSError:
                    pass

                break


_load_pip_cuda_libraries()


class ASREngine:
    ENGINE_LOCAL = "Local (faster-whisper)"
    ENGINE_GOOGLE = "Google Speech API"
    AVAILABLE_ENGINES = [ENGINE_LOCAL, ENGINE_GOOGLE]

    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
    DEVICES = ["auto", "cpu", "cuda"]
    COMPUTE_TYPES = ["auto", "int8", "int8_float16", "int8_float32", "int16", "float16", "float32"]
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
        self.device_index = 0
        self.compute_type = "auto"
        self.cpu_workers = max(1, os.cpu_count() or 1)
        self.gpu_workers_per_device = 1
        self._model = None
        self._batched_pipeline = None

    def configure(
        self,
        engine,
        model_size="base",
        language="auto",
        device="auto",
        compute_type="auto",
        cpu_workers=None,
        gpu_workers_per_device=None,
        device_index=0,
    ):
        needs_reload = (
            self.engine != engine
            or self.model_size != model_size
            or self.device != device
            or self.compute_type != compute_type
            or self.device_index != device_index
        )
        if needs_reload:
            self._model = None
            self._batched_pipeline = None

        self.engine = engine
        self.model_size = model_size
        self.language = language if language and language != "auto" else None
        self.device = device
        self.compute_type = compute_type
        self.device_index = int(device_index)
        if cpu_workers is not None:
            self.cpu_workers = max(1, int(cpu_workers))
        if gpu_workers_per_device is not None:
            self.gpu_workers_per_device = max(1, int(gpu_workers_per_device))

    @staticmethod
    def detect_cuda_device_indices():
        try:
            import ctranslate2

            count = int(ctranslate2.get_cuda_device_count())
            if count > 0:
                return list(range(count))
        except Exception:
            pass

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            parsed = [entry.strip() for entry in visible.split(",") if entry.strip()]
            if parsed:
                return list(range(len(parsed)))
        return []

    def clone(self):
        clone_engine = ASREngine()
        clone_engine.configure(
            engine=self.engine,
            model_size=self.model_size,
            language=self.language or "auto",
            device=self.device,
            compute_type=self.compute_type,
            cpu_workers=self.cpu_workers,
            gpu_workers_per_device=self.gpu_workers_per_device,
            device_index=self.device_index,
        )
        return clone_engine

    def _cpu_threads_per_worker(self):
        total_cores = max(1, os.cpu_count() or 1)
        return max(1, total_cores // max(1, self.cpu_workers))

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel, BatchedInferencePipeline

            device = self.device
            compute_type = self.compute_type
            self._device_status = ""

            model_kwargs = {}
            if device == "cuda":
                model_kwargs["device_index"] = self.device_index

            if device == "auto":
                # try in a cascade order
                for attempt_device, attempt_ct in [
                    ("cuda", "auto"),
                    ("cuda", "float16"),
                    ("cuda", "float32"),
                    ("cpu", "int8"),
                ]:

                    try:
                        attempt_kwargs = {}
                        if attempt_device == "cuda":
                            attempt_kwargs["device_index"] = self.device_index
                        if attempt_device == "cpu":
                            attempt_kwargs["cpu_threads"] = self._cpu_threads_per_worker()
                        self._model = WhisperModel(
                            self.model_size, device=attempt_device, compute_type=attempt_ct,
                            **attempt_kwargs,
                        )

                        self._device_status = (
                            f"[INFO] Model loaded: device={attempt_device}, compute_type={attempt_ct}"
                        )
                        break

                    except Exception as e:
                        self._device_status = (
                            f"[WARNING] Failed {attempt_device}/{attempt_ct}: {e}"
                        )
                        self._model = None
                        continue

                if self._model is None:
                    raise RuntimeError(
                        "Could not load model on any device. Last error: " + self._device_status
                    )
                
            else:
                if device == "cpu" and compute_type == "auto":
                    compute_type = "int8"

                if device == "cuda":
                    attempt_order = [compute_type]
                    if "auto" not in attempt_order:
                        attempt_order.append("auto")
                    if "float32" not in attempt_order:
                        attempt_order.append("float32")

                    last_error = None
                    for attempt_ct in attempt_order:
                        try:
                            self._model = WhisperModel(
                                self.model_size, device=device, compute_type=attempt_ct,
                                **model_kwargs,
                            )
                            self.compute_type = attempt_ct
                            self._device_status = (
                                f"[INFO] Model loaded: device={device}, compute_type={attempt_ct}"
                            )
                            break
                        except Exception as e:
                            last_error = e
                            self._model = None
                            continue

                    if self._model is None:
                        raise RuntimeError(
                            f"Could not load CUDA model with compute_type={compute_type}. Last error: {last_error}"
                        )

                else:
                    if device == "cpu":
                        model_kwargs["cpu_threads"] = self._cpu_threads_per_worker()
                    self._model = WhisperModel(
                        self.model_size, device=device, compute_type=compute_type,
                        **model_kwargs,
                    )

                    self._device_status = (
                        f"[INFO] Model loaded: device={device}, compute_type={compute_type}"
                    )

            self._batched_pipeline = BatchedInferencePipeline(model=self._model)

    @staticmethod
    def _is_cuda_library_error(error):
        message = str(error).lower()
        cuda_error_markers = [
            "libcublas.so",
            "libcudnn",
            "libcuda.so",
            "cublas",
            "cudnn",
            "cuda",
        ]

        return any(marker in message for marker in cuda_error_markers)

    def _consume_transcribe_result(self, audio_file, name, logs):
        detected_language = None
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

        return transcript, detected_language

    def _activate_cpu_fallback(self, logs):
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        self.device = "cpu"
        self.compute_type = "int8"
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type,
        )
        self._batched_pipeline = BatchedInferencePipeline(model=self._model)
        self._device_status = (
            "[WARNING] CUDA runtime not available. Falling back to CPU/int8 for this run."
        )
        logs.append(self._device_status)

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
            if self._device_status:
                logs.append(self._device_status)

            transcript, detected_language = self._consume_transcribe_result(
                audio_file=audio_file,
                name=name,
                logs=logs,
            )

        except Exception as e:
            if self.device in {"auto", "cuda"} and self._is_cuda_library_error(e):
                logs.append(
                    f"[WARNING] CUDA backend failed for {name}: {e}"
                )

                try:
                    self._activate_cpu_fallback(logs)
                    transcript, detected_language = self._consume_transcribe_result(
                        audio_file=audio_file,
                        name=name,
                        logs=logs,
                    )

                    return transcript, detected_language, logs
                
                except Exception as cpu_fallback_error:
                    logs.append(
                        f"[ERROR] CPU fallback also failed for {name}: {cpu_fallback_error}"
                    )

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

    def _build_asr_workers(self, total_files):
        if self.asr.engine != ASREngine.ENGINE_LOCAL:
            worker_count = min(max(1, self.asr.cpu_workers), max(1, total_files))
            workers = [self.asr.clone() for _ in range(worker_count)]
            return workers, f"[DEBUG] Using {worker_count} API worker(s)."

        cuda_indices = ASREngine.detect_cuda_device_indices()
        workers = []

        if self.asr.device in {"auto", "cuda"} and cuda_indices:
            for gpu_index in cuda_indices:
                for _ in range(self.asr.gpu_workers_per_device):
                    worker = self.asr.clone()
                    worker.configure(
                        engine=worker.engine,
                        model_size=worker.model_size,
                        language=worker.language or "auto",
                        device="cuda",
                        compute_type=worker.compute_type,
                        cpu_workers=worker.cpu_workers,
                        gpu_workers_per_device=worker.gpu_workers_per_device,
                        device_index=gpu_index,
                    )
                    workers.append(worker)

            if workers:
                max_workers = min(len(workers), max(1, total_files))
                workers = workers[:max_workers]
                return workers, (
                    f"[DEBUG] Using {len(workers)} local ASR worker(s) on CUDA device(s): {cuda_indices} "
                    f"(workers/GPU={self.asr.gpu_workers_per_device})."
                )

        worker_count = min(max(1, self.asr.cpu_workers), max(1, total_files))
        for _ in range(worker_count):
            worker = self.asr.clone()
            worker.configure(
                engine=worker.engine,
                model_size=worker.model_size,
                language=worker.language or "auto",
                device="cpu" if self.asr.device == "auto" else worker.device,
                compute_type="int8" if self.asr.device == "auto" else worker.compute_type,
                cpu_workers=worker.cpu_workers,
                gpu_workers_per_device=worker.gpu_workers_per_device,
                device_index=0,
            )
            workers.append(worker)

        return workers, f"[DEBUG] Using {worker_count} local ASR worker(s) on CPU."

    def process_wav_files(self, input_dir=None, separator='|'):
        directory = Path(input_dir) if input_dir else self.input_dir
        if not directory.exists():
            yield f"[ERROR] Input directory not found: {directory}"
            return

        wav_files = self._get_wav_files(directory)
        yield f"[DEBUG] Found {len(wav_files)} .wav files. Using '{separator}' as separator."
        yield f"[DEBUG] ASR: {self.asr.engine} | Model: {self.asr.model_size} | Language: {self.asr.language or 'auto-detect'}"

        if not wav_files:
            yield "[WARNING] No .wav files found to transcribe."
            return

        workers, worker_msg = self._build_asr_workers(total_files=len(wav_files))
        yield worker_msg

        metadata = []
        languages_detected = set()
        failed_count = 0
        placeholder = "**NO TRANSCRIPT AVAILABLE, EDIT MANUALLY**"

        results = [None] * len(wav_files)
        executors = [ThreadPoolExecutor(max_workers=1) for _ in workers]
        try:
            future_map = {}
            for idx, wav_file in enumerate(wav_files):
                worker_idx = idx % len(workers)
                yield f"[DEBUG] Queueing: {wav_file.name} -> worker-{worker_idx + 1}"
                future = executors[worker_idx].submit(workers[worker_idx].transcribe, wav_file)
                future_map[future] = (idx, wav_file, worker_idx)

            for future in as_completed(future_map):
                idx, wav_file, worker_idx = future_map[future]
                try:
                    transcript, detected_lang, transcribe_logs = future.result()
                except Exception as e:
                    transcript = placeholder
                    detected_lang = None
                    transcribe_logs = [f"[ERROR] Worker-{worker_idx + 1} failed for {wav_file.name}: {e}"]

                for log in transcribe_logs:
                    yield f"[worker-{worker_idx + 1}] {log}"
                results[idx] = (wav_file, transcript, detected_lang)
        finally:
            for executor in executors:
                executor.shutdown(wait=True)

        for result in results:
            wav_file, transcript, detected_lang = result
            if detected_lang:
                languages_detected.add(detected_lang)
            if transcript == placeholder:
                failed_count += 1
            metadata.append([Path('wavs', wav_file.name).as_posix(), transcript])

        yield "[DEBUG] Writing metadata.csv..."
        df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
        df.to_csv(self.metadata_file, sep=separator, index=False, header=False)

        lang_summary = ", ".join(sorted(languages_detected)) if languages_detected else "N/A"
        yield f"[DEBUG] Languages detected: {lang_summary}"

        total = len(metadata)
        succeeded = total - failed_count
        if failed_count == 0:
            yield f"[DEBUG] metadata.csv generated successfully\n\n[OK] Finished processing {total} files."
        
        elif failed_count == total:
            yield f"[ERROR] All {total} files failed transcription. metadata.csv was written with placeholders.\n\n[FAIL] No usable transcripts were generated. Check your ASR settings and audio files."
        
        else:
            yield f"[WARNING] {failed_count}/{total} files failed transcription (placeholders written).\n\n[OK] Finished processing {total} files ({succeeded} succeeded, {failed_count} failed)."

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

    def gradio_run(self, separator, asr_engine, model_size, language, device, compute_type, cpu_workers, gpu_workers_per_device):
        if not check_wav_files():
            yield "ERROR: No .wav files found in the input directory. Please upload them and try again."
            return

        self.asr.configure(
            engine=asr_engine,
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
            cpu_workers=cpu_workers,
            gpu_workers_per_device=gpu_workers_per_device,
        )

        logs = []
        for log in self.process_wav_files(separator=separator):
            logs.append(log)
            yield "\n".join(logs)