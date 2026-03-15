import os
import pandas as pd
import gradio as gr
import shutil
from pathlib import Path
from pydub import AudioSegment

from functions.filter import NoiseReducer
from functions.split import AudioSplitter
from functions.main import MainProcess, ASREngine
from functions.sanitycheck import SanityChecker

from functions.helper.janitor import Janitor

class LJSpeechDatasetUI:
    def __init__(self, dataset_dir, metadata_file):
        self.dataset_dir = str(Path(dataset_dir))
        self.metadata_file = str(Path(metadata_file))
        self.separator = '|'
        self.min_duration = 4000 #both in miliseconds
        self.max_duration = 10000
        self.metadata = self._load_metadata()

        #denoiser values
        self.frame_length = 2048
        self.hop_length = 512
        self.silence_threshold = 0.1
        self.noise_reduction_strength = 0.6
        self.use_spectral_gating = False  #pytorch spectral gating

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                df = pd.read_csv(self.metadata_file, sep=self.separator if self.separator else '|', header=None, names=["wav_filename", "transcript", "normalized_transcript"], dtype=str)
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=["filename", "transcript"])
            df["filename"] = df["wav_filename"].astype(str)
            df = df[["filename", "transcript"]]
            return df
        return pd.DataFrame(columns=["filename", "transcript"])

    def load_data(self):
        if not os.path.exists(self.dataset_dir):
            return []

        audio_files = sorted(f for f in os.listdir(self.dataset_dir) if f.lower().endswith(".wav"))
        if len(audio_files) == 0:
            return []

        if isinstance(self.metadata, pd.DataFrame) and not self.metadata.empty:
            meta_map = {os.path.basename(r["filename"]): r["transcript"] for _, r in self.metadata.iterrows()}
        else:
            meta_map = {}

        return [(os.path.join(self.dataset_dir, af), meta_map.get(af, "")) for af in audio_files]

    def update_metadata(self, audio_file, new_transcript):
        base_name = os.path.basename(audio_file)
        mask = self.metadata["filename"].apply(os.path.basename) == base_name
        if mask.any():
            self.metadata.loc[mask, "transcript"] = new_transcript
        else:
            new_row = pd.DataFrame({"filename": [Path("wavs", base_name).as_posix()], "transcript": [new_transcript]})
            self.metadata = pd.concat([self.metadata, new_row], ignore_index=True)
        self.metadata.to_csv(self.metadata_file, sep=self.separator if self.separator else '|', index=False, header=False)
        return f"Transcript for {base_name} updated: {new_transcript}"

    def create_interface(self):
        #constants
        items_per_page = 10  #number of items to display per page

        def save_uploaded_files(file_paths, auto_convert_mp3):
            if not file_paths:
                return "No files uploaded."

            os.makedirs(self.dataset_dir, exist_ok=True)
            allowed_ext = {".wav", ".mp3"} if auto_convert_mp3 else {".wav"}
            converted = 0
            copied = 0

            try:
                dataset_dir_resolved = Path(self.dataset_dir).resolve()
                for temp_file_path in file_paths:
                    file_name = os.path.basename(temp_file_path)
                    # stays strictly inside dataset_dir before any file operation
                    if (dataset_dir_resolved / file_name).resolve().parent != dataset_dir_resolved:
                        return f"Invalid file name: {file_name}"
                    
                    ext = Path(file_name).suffix.lower()

                    if ext not in allowed_ext:
                        fmt = ".wav and .mp3" if auto_convert_mp3 else ".wav"
                        return f"Only {fmt} files are allowed."

                    if ext == ".mp3" and auto_convert_mp3:
                        wav_name = Path(file_name).stem + ".wav"
                        dest_file_path = os.path.join(self.dataset_dir, wav_name)
                        audio = AudioSegment.from_mp3(temp_file_path)
                        audio.export(dest_file_path, format="wav")
                        converted += 1
                    else:
                        dest_file_path = os.path.join(self.dataset_dir, file_name)
                        shutil.copy(temp_file_path, dest_file_path)
                        copied += 1

                parts = []
                if copied:
                    parts.append(f"{copied} .wav file(s) uploaded")
                if converted:
                    parts.append(f"{converted} .mp3 file(s) converted to .wav")
                return ". ".join(parts) + "." if parts else "No files processed."
            except Exception as e:
                return f"Error: {e}"

        def refresh_data(current_page):
            self.metadata = self._load_metadata()
            data = self.load_data()
            total_items = len(data)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            if current_page < 1:
                current_page = 1
            elif current_page > total_pages:
                current_page = total_pages

            start_idx = (current_page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_data = data[start_idx:end_idx]

            updates = []
            file_paths = []

            for i in range(items_per_page):
                if i < len(page_data):
                    audio_file, transcript = page_data[i]
                    audio_update = gr.update(value=audio_file, visible=True)
                    transcript_update = gr.update(value=transcript, visible=True)
                    save_btn_update = gr.update(visible=True)
                    status_box_update = gr.update(value="", visible=True)
                    file_paths.append(audio_file)
                else:
                    audio_update = gr.update(visible=False)
                    transcript_update = gr.update(value="", visible=False)
                    save_btn_update = gr.update(visible=False)
                    status_box_update = gr.update(value="", visible=False)
                    file_paths.append(None)
                updates.extend([audio_update, transcript_update, save_btn_update, status_box_update])

            #update page box
            page_label_update = f"Page {current_page} of {total_pages}"

            return updates + [file_paths, current_page, page_label_update]

        def save_transcript(index):
            def inner(transcript, file_paths):
                audio_file = file_paths[index]
                if audio_file is None:
                    return "No file loaded"
                result = self.update_metadata(audio_file, transcript)
                return result
            return inner
        
        def update_file_list():
            dataset_path = Path(self.dataset_dir)
            if not dataset_path.exists():
                return ""
            audio_files = sorted(f.name for f in dataset_path.iterdir() if f.suffix.lower() == '.wav')
            return "\n".join(audio_files)
        
        def handle_upload(file_paths, auto_convert_mp3):
            status = save_uploaded_files(file_paths, auto_convert_mp3)
            file_list_content = update_file_list()
            return status, file_list_content

        with gr.Blocks(title="LJSpeech Dataset Generator", theme=gr.themes.Citrus()) as app:
            gr.Markdown("<div style='text-align: center;'><h1>LJSpeech Dataset Generator</h1></div>")

            noise_reducer = NoiseReducer()
            splitter = AudioSplitter()
            main_process = MainProcess(input_dir=self.dataset_dir, metadata_file=self.metadata_file)
            sanitycheck = SanityChecker(metadata_file=self.metadata_file, wav_directory=self.dataset_dir)

            with gr.Tab("File Upload"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Upload your audio files here for pre-processing.**")
                        auto_convert_mp3 = gr.Checkbox(
                            label="Auto-convert MP3 to WAV",
                            info="When enabled, .mp3 files are accepted and automatically converted to .wav on upload.",
                            value=False,
                        )
                        upload_audio = gr.File(label="Upload audio files", file_types=["audio"], file_count="multiple", type="filepath")
                        upload_status = gr.Textbox(label="Output", interactive=False)
                        
                    with gr.Column():
                        gr.Markdown("**Uploaded files will be displayed here.**")
                        file_list = gr.Textbox(label="Uploaded Files", lines=6, interactive=False)
                        file_list_update = gr.Button("Update", variant="primary")


                upload_audio.upload(handle_upload, inputs=[upload_audio, auto_convert_mp3], outputs=[upload_status, file_list])
                file_list_update.click(update_file_list, inputs=[], outputs=file_list)

            with gr.Tab("Pre-Processing"):
                with gr.Row():
                    
                    with gr.Column():
                        pp_chunk = gr.Button("Step 1 - Chunking", variant="primary")
                        gr.Markdown("Chunking is the process of splitting a long audio file into smaller portions. This is recommended.")

                    with gr.Column():
                        pp_filter = gr.Button("Step 2 - Filter Background Noise", variant="stop")
                        gr.Markdown("Filters background noise while retaining crisp speech. Uses single-pass noise reduction with automatic noise profiling. Enable spectral gating for automatic processing, or adjust parameters manually.")

                    with gr.Column():
                        pp_main = gr.Button("Step 3 - Auto Transcript", variant="primary")
                        gr.Markdown("The final step or preprocessing. This will generate the metadata.csv file.")

                with gr.Row():

                    with gr.Column():
                        gr.Markdown("**Denoiser Controls**")
                        frame_length = gr.Slider(
                            label="Frame Length",
                            minimum=256,
                            maximum=8192,
                            step=128,
                            value=self.frame_length
                        )

                        hop_length = gr.Slider(
                            label="Hop Length",
                            minimum=64,
                            maximum=4096,
                            step=64,
                            value=self.hop_length
                        )

                        silence_threshold = gr.Slider(
                            label="Silence Threshold",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=self.silence_threshold
                        )

                        noise_reduction_strength = gr.Slider(
                            label="Noise Reduction Strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=self.noise_reduction_strength
                        )

                        use_spectral_gating = gr.Checkbox(
                            label="Use Spectral Gating",
                            info="Enable spectral gating for automatic noise profiling. Overrides manual parameters above.",
                            value=self.use_spectral_gating
                        )

                        save_denoiser = gr.Button("Save", variant="secondary")

                    with gr.Column():
                        gr.Markdown("**Chunking Controls**")
                        min_duration_slider = gr.Slider(
                            label="Minimum Chunking Duration (ms)",
                            minimum=1000,
                            maximum=10000,
                            step=500,
                            value=self.min_duration
                        )

                        max_duration_slider = gr.Slider(
                            label="Maximum Chunking Duration (ms)",
                            minimum=1000,
                            maximum=20000,
                            step=500,
                            value=self.max_duration
                        )

                        save_dur = gr.Button("Save", variant="secondary")
                    
                        gr.Markdown("**Separator Controls**")
                        separator_val = gr.Textbox(label="Separator", value="|", interactive=True)
                        save_sep = gr.Button("Save", variant="secondary")

                        gr.Markdown("**ASR Settings**")
                        asr_engine = gr.Dropdown(
                            label="ASR Engine",
                            choices=ASREngine.AVAILABLE_ENGINES,
                            value=ASREngine.AVAILABLE_ENGINES[0],
                        )
                        asr_model_size = gr.Dropdown(
                            label="Model Size (local only)",
                            choices=ASREngine.MODEL_SIZES,
                            value="base",
                        )
                        asr_language = gr.Dropdown(
                            label="Language",
                            choices=ASREngine.LANGUAGES,
                            value="auto",
                        )
                        asr_device = gr.Dropdown(
                            label="Device (local only)",
                            choices=ASREngine.DEVICES,
                            value="auto",
                        )
                        asr_compute_type = gr.Dropdown(
                            label="Compute Type (local only)",
                            choices=ASREngine.COMPUTE_TYPES,
                            value="auto",
                        )
                        asr_cpu_workers = gr.Slider(
                            label="CPU ASR Workers",
                            minimum=1,
                            maximum=max(1, os.cpu_count() or 1),
                            step=1,
                            value=min(4, max(1, os.cpu_count() or 1)),
                        )
                        asr_gpu_workers_per_device = gr.Slider(
                            label="GPU Workers Per Device",
                            minimum=1,
                            maximum=4,
                            step=1,
                            value=1,
                        )
                        
                    with gr.Column():
                        gr.Markdown("**Settings Status**")
                        settings_update = gr.Textbox(label="Output", lines=4, interactive=False)
                        settings_curr = gr.Textbox(
                            label="Current Settings",
                            value=f"""Denoiser Settings:\nFrame Length: {self.frame_length}\nHop Length: {self.hop_length}\nSilence Threshold: {self.silence_threshold}\nNoise Reduction Strength: {self.noise_reduction_strength}\nUse Spectral Gating: {self.use_spectral_gating}\n\nChunking Duration:\nMinimum: {self.min_duration} ms | Maximum: {self.max_duration} ms\n\nSeparator:\n{self.separator}""",
                            lines=12, interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Settings Information**")
                        gr.Markdown("""**Denoiser:**
Parameter adjustments to filter background noise.

**Chunking:**
Adjust the minimum and maximum duration for splitting audio files. Splits at silence boundaries to avoid cutting mid-speech.

**Separator:**
Adjust the separator character for metadata.csv

**ASR Settings:**
Configure the speech recognition engine.

- **Engine**: Local (faster-whisper) is recommended for speed and offline use. Google Speech API is a remote fallback. Model is downloaded on first use.
- **Model Size**: Larger models are more accurate but slower and use more memory.
  - `tiny` — Fastest, least accurate
  - `base` — Good balance (default)
  - `small` / `medium` — Better accuracy
  - `large-v3` — Best accuracy, slowest
- **Language**: Set to `auto` for automatic detection, or pick a specific language code.
- **Device**: Where the model runs.
  - `auto` — Tries CUDA first, falls back to CPU
  - `cpu` — Force CPU (always works)
  - `cuda` — Force NVIDIA GPU (requires CUDA)
- **Compute Type**: Precision for model inference.
  - `auto` — Let the engine decide
  - `int8` — Fastest on CPU, recommended for CPU-only
  - `int8_float16` — Mixed precision (GPU, compute capability ≥ 6.1)
  - `int8_float32` — Mixed precision (broader GPU support)
  - `int16` — Half-integer precision
  - `float16` — Half-float (GPU only)
  - `float32` — Full precision, safest fallback
- **CPU ASR Workers**: Number of concurrent transcription workers when running on CPU.
- **GPU Workers Per Device**: Number of concurrent workers per visible CUDA GPU.

**You should leave all of these options alone if you don't understand them.**""")

                pp_status = gr.Textbox(label="Output", lines=10, interactive=False)
            
                pp_filter.click(noise_reducer.gradio_run, inputs=[frame_length, hop_length, silence_threshold, noise_reduction_strength, use_spectral_gating], outputs=pp_status)
                pp_chunk.click(splitter.gradio_run, inputs=[min_duration_slider, max_duration_slider], outputs=pp_status)
                pp_main.click(
                    main_process.gradio_run,
                    inputs=[
                        separator_val,
                        asr_engine,
                        asr_model_size,
                        asr_language,
                        asr_device,
                        asr_compute_type,
                        asr_cpu_workers,
                        asr_gpu_workers_per_device,
                    ],
                    outputs=pp_status,
                )
    
                def update_separator(new_sep):
                    if not new_sep or len(new_sep) != 1:
                        return "Separator must be a single character."
                    self.separator = new_sep
                    return f"Separator updated to: \n{new_sep}"
                
                def update_duration(min_duration, max_duration):
                    if min_duration > max_duration:
                        return "Minimum duration cannot be greater than maximum duration."
                    self.min_duration = min_duration
                    self.max_duration = max_duration
                    return f"Duration updated to: \n{min_duration}ms (min) \n{max_duration}ms (max)"
                
                def update_denoiser(frame_length, hop_length, silence_threshold, noise_reduction_strength, use_spectral_gating):
                    self.frame_length = frame_length
                    self.hop_length = hop_length
                    self.silence_threshold = silence_threshold
                    self.noise_reduction_strength = noise_reduction_strength
                    self.use_spectral_gating = use_spectral_gating

                    return f"Denoiser settings updated: \nFrame Length: {self.frame_length}, \nHop Length: {self.hop_length}, \nSilence Threshold: {self.silence_threshold}, \nNoise Reduction Strength: {self.noise_reduction_strength}, \nUse Spectral Gating: {self.use_spectral_gating}"

                def update_settings_display():
                    return f"""Denoiser Settings:\nFrame Length: {self.frame_length}\nHop Length: {self.hop_length}\nSilence Threshold: {self.silence_threshold}\nNoise Reduction Strength: {self.noise_reduction_strength}\nUse Spectral Gating: {self.use_spectral_gating}\n\nChunking Duration:\nMinimum: {self.min_duration} ms | Maximum: {self.max_duration} ms\n\nSeparator:\n{self.separator}"""

                save_sep.click(update_separator, inputs=[separator_val], outputs=settings_update)
                save_dur.click(update_duration, inputs=[min_duration_slider, max_duration_slider], outputs=settings_update)
                save_denoiser.click(update_denoiser, inputs=[frame_length, hop_length, silence_threshold, noise_reduction_strength, use_spectral_gating], outputs=settings_update)
                
                save_sep.click(fn=update_settings_display, inputs=[], outputs=settings_curr)
                save_dur.click(fn=update_settings_display, inputs=[], outputs=settings_curr)
                save_denoiser.click(fn=update_settings_display, inputs=[], outputs=settings_curr)

            with gr.Tab("Transcript Editing"):
                components = []

                #dft states
                file_states = gr.State(value=[None] * items_per_page)
                current_page_state = gr.State(value=1)
                page_label_state = gr.State(value="Page 1 of 1")

                #dfine elements on top
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Data", variant="primary")
                    previous_btn = gr.Button("Previous", variant="secondary")
                    next_btn = gr.Button("Next", variant="secondary")
                    page_label = gr.Markdown("Page 1 of 1")

                #template for creating components
                for i in range(0, items_per_page, 4):
                    with gr.Row():
                        for j in range(4):
                            if i + j < items_per_page:
                                with gr.Column():
                                    audio_component = gr.Audio(visible=False)
                                    transcript_box = gr.Textbox(visible=False, label="Transcript", lines=3, interactive=True)
                                    save_btn = gr.Button("Update", visible=False)
                                    status_box = gr.Textbox(visible=False, label="Status", interactive=False)
                                    components.append((audio_component, transcript_box, save_btn, status_box))

                #save outputs
                outputs = []
                for component in components:
                    outputs.extend(component)

                outputs.extend([file_states, current_page_state, page_label])
                refresh_btn.click(fn=lambda: refresh_data(1), inputs=[], outputs=outputs)
                previous_btn.click(fn=lambda current_page: refresh_data(current_page - 1), inputs=[current_page_state], outputs=outputs)
                next_btn.click(fn=lambda current_page: refresh_data(current_page + 1), inputs=[current_page_state], outputs=outputs)

                for index, (audio_component, transcript_box, save_btn, status_box) in enumerate(components):
                    save_btn.click(save_transcript(index), inputs=[transcript_box, file_states], outputs=status_box)

            with gr.Tab("Post Processing"):
                with gr.Row():
                    with gr.Column():
                        san_check = gr.Button("Step 1: Sanity Check", variant="primary")
                        package_data = gr.Button("Step 2: Package Dataset", variant="primary")

                    with gr.Column():
                        san_status = gr.Textbox(label="Sanity Check Output", interactive=False)
                        download_link = gr.File(label="Download Packaged Dataset", interactive=False)

                san_check.click(sanitycheck.run_check, inputs=[], outputs=san_status)
                package_data.click(main_process.zip_output, inputs=[], outputs=[download_link])

            with gr.Tab("Cleanup"):
                with gr.Row():

                    with gr.Column():
                        gr.Markdown("**This button resets the dataset files. Use with caution.**")
                        cleanup_btn = gr.Button("Cleanup", variant="secondary")

                    with gr.Column():
                        cleanup_output = gr.Textbox(label="Cleanup Output", lines=5, interactive=False)

                cleanup_btn.click(Janitor.reset_dataset_files, inputs=[], outputs=cleanup_output)
            
            with gr.Tab("Instructions"):

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("# Instructions")
                        gr.Markdown("_Documentation for all available features_")

                        with gr.Accordion("File Upload", open=False):
                            gr.Markdown("""
                            1. Navigate to the `File Upload` tab
                            2. Enable "Auto-convert MP3 to WAV" if you have .mp3 files
                            3. Upload your .wav (or .mp3) audio files 
                            4. Files will appear in the list automatically
                            5. Use the refresh button if needed to update the list
                            """)

                        with gr.Accordion("Pre-Processing", open=False):
                            gr.Markdown("### Step 1: Audio Chunking")
                            gr.Markdown("""
                            - Required step to split long audio files into smaller segments
                            - Splits at silence boundaries to avoid cutting mid-speech
                            - Configurable settings:
                                - Minimum duration (ms)
                                - Maximum duration (ms)
                            - Click "Step 1 - Chunking" to begin processing
                            """)
                            
                            gr.Markdown("### Step 2: Noise Filtering")
                            gr.Markdown("""
                            - Optional step to improve audio quality
                            - Single-pass noise reduction with automatic noise profiling
                            - Adjustable parameters:
                                - Frame Length & Hop Length
                                - Silence Threshold
                                - Noise Reduction Strength
                            - Enable Spectral Gating for fully automatic processing
                            """)
                            
                            gr.Markdown("### Step 3: Auto Transcription") 
                            gr.Markdown("""
                            - Generates metadata.csv containing transcripts
                            - Configurable ASR engine:
                                - **Local (faster-whisper)**: Fast, offline, auto language detection
                                - **Google Speech API**: Remote fallback (requires internet)
                            - Adjustable model size, language, and compute device
                            - Configurable CSV separator character
                            - Model is downloaded automatically on first use
                            """)

                        with gr.Accordion("Transcript Editing", open=False):
                            gr.Markdown("""
                            - Review and edit auto-generated transcripts
                            - Features:
                                - Audio playback
                                - Text editing
                                - Pagination controls
                            - Click "Update" after editing to save changes
                            """)

                        with gr.Accordion("Post Processing", open=False):
                            gr.Markdown("### Sanity Check")
                            gr.Markdown("""
                            - Validates metadata.csv file
                            - Checks for:
                                - Missing transcripts
                                - Incorrect number of files/lines
                            - Recommended before packaging
                            """)
                            
                            gr.Markdown("### Package Dataset")
                            gr.Markdown("""
                            - Creates final ZIP archive containing:
                                - Processed audio files
                                - metadata.csv
                                - All required dataset files
                            """)

                        with gr.Accordion("Cleanup", open=False):
                            gr.Markdown("""
                            ⚠️ **Warning: Destructive Operation**
                            
                            - Removes all:
                                - Input audio files
                                - Generated chunks
                                - Processed files
                                - metadata.csv
                            - Use only when starting a completely new dataset
                            - This action cannot be undone
                            """)

                        gr.Markdown("---")
                        gr.Markdown("_Documentation will be updated as new features are added_")
                    
                    with gr.Column():
                        gr.Markdown("# Alpaca")
                        gr.Textbox(
                            label="Alpaca!", 
                            lines=10, 
                            value=r"""
                    /\⌒⌒⌒/\
                    (⦿('◞◟')⦿)
                    (            )
                    (            )       ◿
                    (                    )
                    (____________)
                        ◤          ◤
                            
                    yes its out of shape""",
                            interactive=False
                        )

            gr.Markdown("<div style='text-align: center;'>Something doesn't work? Feel free to open an issue on <a href='https://github.com/DominicTWHV/LJSpeech_Dataset_Generator'>GitHub</a></div>")
            gr.Markdown("<div style='text-align: center;'>Built by Dominic with ❤️</div>")

        return app

if __name__ == "__main__":

    ui = LJSpeechDatasetUI(dataset_dir="wavs", metadata_file="metadata.csv")
    app = ui.create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)