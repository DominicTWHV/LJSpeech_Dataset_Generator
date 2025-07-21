import os
import pandas as pd
import gradio as gr
import shutil

from functions.filter import NoiseReducer
from functions.split import AudioSplitter
from functions.main import MainProcess
from functions.sanitycheck import SanityChecker

from functions.helper.janitor import Janitor

class LJSpeechDatasetUI:
    def __init__(self, dataset_dir, metadata_file):
        self.dataset_dir = dataset_dir
        self.metadata_file = metadata_file
        self.separator = '|'
        self.min_duration = 4000 #both in miliseconds
        self.max_duration = 10000
        self.metadata = self._load_metadata()

        #denoiser values
        self.frame_length = 2048
        self.hop_length = 512
        self.silence_threshold = 0.1
        self.prop_decrease_noisy = 0.9
        self.prop_decrease_normal = 0.3
        self.use_spectral_gating = False  #pytorch spectral gating

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            #read csv
            df = pd.read_csv(self.metadata_file, sep=self.separator if self.separator else '|', header=None,names=["wav_filename", "transcript", "normalized_transcript"], dtype=str,)
            df["filename"] = df["wav_filename"].astype(str)
            df = df[["filename", "transcript"]]
            return df
        else:
            #return an empty DataFrame with specified columns
            df = pd.DataFrame(columns=["filename", "transcript"])
            return df

    def load_data(self):
        try:
            if not os.path.exists(self.dataset_dir):
                return []
            
            audio_files = [f for f in os.listdir(self.dataset_dir) if f.endswith(".wav")]
            if len(audio_files) == 0:
                return []
            
            if isinstance(self.metadata, pd.DataFrame) and not self.metadata.empty:
                meta_map = {os.path.basename(r["filename"]): r["transcript"] for _, r in self.metadata.iterrows()}
            else:
                meta_map = {}
            #Debug statements
            #print("Audio files:", audio_files)
            #print("Metadata filenames:", self.metadata["filename"].tolist())
            #print("Meta map:", meta_map)
            data = [(os.path.join(self.dataset_dir, af), meta_map.get(af, "")) for af in audio_files]
            #print("Data:", data)
            return data
        except Exception as e:
            return []

    def update_metadata(self, audio_file, new_transcript):
        base_name = os.path.basename(audio_file)
        mask = self.metadata["filename"].apply(os.path.basename) == base_name
        if mask.any():
            self.metadata.loc[mask, "transcript"] = new_transcript
        else:
            new_row = pd.DataFrame({"filename": [base_name], "transcript": [new_transcript]})
            self.metadata = pd.concat([self.metadata, new_row], ignore_index=True)
        self.metadata.to_csv(self.metadata_file, sep=self.separator if self.separator else '|', index=False, header=False)
        return f"Transcript for {base_name} updated: {new_transcript}"

    def create_interface(self):
        #constants
        items_per_page = 10  #number of items to display per page

        def save_uploaded_files(file_paths):
            if not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)

            try:
                for temp_file_path in file_paths:
                    file_name = os.path.basename(temp_file_path)
                    
                    if not file_name.endswith(".wav"):
                        return "Only .wav files are allowed."

                    dest_file_path = os.path.join(self.dataset_dir, file_name)
                    shutil.copy(temp_file_path, dest_file_path)

                return f"{len(file_paths)} file(s) uploaded successfully."
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
            if not os.path.exists(self.dataset_dir):
                return ""
            audio_files = [f for f in os.listdir(self.dataset_dir) if f.endswith(".wav")]
            return "\n".join(audio_files)
        
        def handle_upload(file_paths):
                    status = save_uploaded_files(file_paths)
                    file_list_content = update_file_list()
                    return status, file_list_content

        with gr.Blocks(title="LJSpeech Dataset Generator", theme=gr.themes.Citrus()) as app:
            gr.Markdown("<div style='text-align: center;'><h1>LJSpeech Dataset Generator</h1></div>")

            noise_reducer = NoiseReducer()
            splitter = AudioSplitter()
            main_process = MainProcess()
            sanitycheck = SanityChecker()

            with gr.Tab("File Upload"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Upload your .wav files here for pre-processing.**")
                        upload_audio = gr.File(label="Upload .wav files", file_types=["audio"], file_count="multiple", type="filepath")
                        upload_status = gr.Textbox(label="Output", interactive=False)
                        
                    with gr.Column():
                        gr.Markdown("**Uploaded files will be displayed here.**")
                        file_list = gr.Textbox(label="Uploaded Files", lines=6, interactive=False)
                        file_list_update = gr.Button("Update", variant="primary")


                upload_audio.upload(handle_upload, inputs=upload_audio, outputs=[upload_status, file_list])
                file_list_update.click(update_file_list, inputs=[], outputs=file_list)

            with gr.Tab("Pre-Processing"):
                with gr.Row():
                    
                    with gr.Column():
                        pp_chunk = gr.Button("Step 1 - Chunking", variant="primary")
                        gr.Markdown("Chunking is the process of splitting a long audio file into smaller portions. This is recommended.")

                    with gr.Column():
                        pp_filter = gr.Button("Step 2 - Filter Background Noise", variant="stop")
                        gr.Markdown("_The noise filtering function is in beta and may cause issues. Use with caution, or skip directly to the next step._")
                    
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

                        prop_decrease_noisy = gr.Slider(
                            label="Propagate Decrease (Noisy)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=self.prop_decrease_noisy
                        )

                        prop_decrease_normal = gr.Slider(
                            label="Propagate Decrease (Normal)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=self.prop_decrease_normal
                        )

                        use_spectral_gating = gr.Checkbox(
                            label="Use Spectral Gating",
                            info="Enable this if you have enough compute power to use PyTorch Spectral Gating. This option will deem all other parameters above useless.",
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
                    
                        gr.Markdown("**Seperator Controls**")
                        separator_val = gr.Textbox(label="Separator", value="|", interactive=True)
                        save_sep = gr.Button("Save", variant="secondary")
                        
                    with gr.Column():
                        gr.Markdown("**Settings Status**")
                        settings_update = gr.Textbox(label="Output", lines=4, interactive=False)
                        settings_curr = gr.Textbox(
                            label="Current Settings",
                            value=f"""Denoiser Settings:\nFrame Length: {self.frame_length}\nHop Length: {self.hop_length}\nSilence Threshold: {self.silence_threshold}\nPropagate Decrease (Noisy): {self.prop_decrease_noisy}\nPropagate Decrease (Normal): {self.prop_decrease_normal}\nUse Spectral Gating: {self.use_spectral_gating}\n\nChunking Duration:\nMinimum: {self.min_duration} ms | Maximum: {self.max_duration} ms\n\nSeparator:\n{self.separator}""",
                            lines=12, interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Settings Information**")
                        gr.Markdown("Denoiser:\nParameter adjustments to filter background noise.\n\nChunking:\nAdjust the minimum and maximum duration for splitting audio files.\n\nSeparator:\nAdjust the separator for metadata.csv\n\n\n**You should leave all of these options alone if you don't understand these.**")

                pp_status = gr.Textbox(label="Output", lines=10, interactive=False)
            
                pp_filter.click(noise_reducer.gradio_run, inputs=[frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal, use_spectral_gating], outputs=pp_status)
                pp_chunk.click(splitter.gradio_run, inputs=[min_duration_slider, max_duration_slider], outputs=pp_status)
                pp_main.click(main_process.gradio_run, inputs=[separator_val], outputs=pp_status)
    
                def update_separator(new_sep):
                    self.separator = new_sep
                    return f"Separator updated to: \n{new_sep}"
                
                def update_duration(min_duration, max_duration):
                    self.min_duration = min_duration
                    self.max_duration = max_duration
                    return f"Duration updated to: \n{min_duration}ms (min) \n{max_duration}ms (max)"
                
                def update_denoiser(frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal):
                    self.frame_length = frame_length
                    self.hop_length = hop_length
                    self.silence_threshold = silence_threshold
                    self.prop_decrease_noisy = prop_decrease_noisy
                    self.prop_decrease_normal = prop_decrease_normal

                    return f"Denoiser settings updated: \nFrame Length: {self.frame_length}, \nHop Length: {self.hop_length}, \nSilence Threshold: {self.silence_threshold}, \nPropagate Decrease (Noisy): {self.prop_decrease_noisy}, \nPropagate Decrease (Normal): {self.prop_decrease_normal}"
    
                def update_settings_display():
                    return f"""Denoiser Settings:\nFrame Length: {self.frame_length}\nHop Length: {self.hop_length}\nSilence Threshold: {self.silence_threshold}\nPropagate Decrease (Noisy): {self.prop_decrease_noisy}\nPropagate Decrease (Normal): {self.prop_decrease_normal}\n\nChunking Duration:\nMinimum: {self.min_duration} ms | Maximum: {self.max_duration} ms\n\nSeparator:\n{self.separator}"""

                save_sep.click(update_separator, inputs=[separator_val], outputs=settings_update)
                save_dur.click(update_duration, inputs=[min_duration_slider, max_duration_slider], outputs=settings_update)
                save_denoiser.click(update_denoiser, inputs=[frame_length, hop_length, silence_threshold, prop_decrease_noisy, prop_decrease_normal], outputs=settings_update)
                
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
                            2. Click to upload your .wav audio files 
                            3. Files will appear in the list automatically
                            4. Use the refresh button if needed to update the list
                            """)

                        with gr.Accordion("Pre-Processing", open=False):
                            gr.Markdown("### Step 1: Audio Chunking")
                            gr.Markdown("""
                            - Required step to split long audio files into smaller segments
                            - Configurable settings:
                                - Minimum duration (ms)
                                - Maximum duration (ms)
                            - Click "Step 1 - Chunking" to begin processing
                            """)
                            
                            gr.Markdown("### Step 2: Noise Filtering (Beta)")
                            gr.Markdown("""
                            - Optional step to improve audio quality
                            - Can be run before or after chunking
                            - Adjustable parameters:
                                - Frame Length
                                - Hop Length 
                                - Silence Threshold
                                - Noise Reduction Levels
                            ⚠️ This feature is in beta - use with caution
                            """)
                            
                            gr.Markdown("### Step 3: Auto Transcription") 
                            gr.Markdown("""
                            - Generates metadata.csv containing transcripts
                            - Uses AI speech recognition
                            - Configurable CSV separator character
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