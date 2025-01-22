import os
import pandas as pd
import gradio as gr
import shutil

from functions.filter import NoiseReducer
from functions.split import AudioSplitter
from functions.main import MainProcess
from functions.sanitycheck import SanityChecker


class LJSpeechDatasetUI:
    def __init__(self, dataset_dir="wavs", metadata_file="metadata.csv"):
        self.dataset_dir = dataset_dir
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        self.separator = '|'  # Ensure separator is initialized

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            # Read CSV
            df = pd.read_csv(
                self.metadata_file,
                sep=self.separator,
                header=None,
                names=["wav_filename", "transcript", "normalized_transcript"],
                dtype=str,
            )
            df["filename"] = df["wav_filename"].astype(str)
            df = df[["filename", "transcript"]]
            return df
        else:
            df = pd.DataFrame(columns=["filename", "transcript"])
            return df

    def load_data(self):
        try:
            if not os.listdir(self.dataset_dir):
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
                duplicate_files = []
                for temp_file_path in file_paths:
                    file_name = os.path.basename(temp_file_path)
                    dest_file_path = os.path.join(self.dataset_dir, file_name)
                    if os.path.exists(dest_file_path):
                        duplicate_files.append(file_name)
                
                if duplicate_files:
                    return "Error: The following file(s) already exist:\n" + "\n ".join(duplicate_files)

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
            print(f"Metadata loaded: {self.metadata}")
            data = self.load_data()
            print(f"Data loaded: {data}")

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

        with gr.Blocks(title="LJSpeech Dataset Generator", theme=gr.themes.Ocean()) as app:

            gr.Markdown("## LJSpeech Dataset Generator")

            noise_reducer = NoiseReducer()
            splitter = AudioSplitter()
            main_process = MainProcess()
            sanitycheck = SanityChecker()

            with gr.Tab("File Upload"):
                upload_audio = gr.File(label="Upload .wav files", file_types=["audio"], file_count="multiple", type="filepath")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                upload_audio.upload(save_uploaded_files, inputs=upload_audio, outputs=upload_status)

            with gr.Tab("Preprocessing"):
                with gr.Row():
                    pp_filter = gr.Button("Step 1: Preprocess - Filter Background Noise")
                    pp_chunk = gr.Button("Step 2: Preprocess - Chunking")
                    pp_main = gr.Button("Step 3: Preprocess - Auto Transcript")
                
                with gr.Column():
                    self.separator_input = gr.Textbox(label="Separator", value="|", interactive=True)
                    gr.Markdown("Note: You can configure the seperator here. Leave it on default if you do not know what this is. You should only change this if your TTS engine requires a specific seperator.")
            
                pp_status = gr.Textbox(label="Preprocess Status", lines=20, interactive=False)
            
                pp_filter.click(noise_reducer.gradio_run, inputs=[], outputs=pp_status)
                pp_chunk.click(splitter.gradio_run, inputs=[], outputs=pp_status)
                pp_main.click(main_process.gradio_run, inputs=[self.separator_input], outputs=pp_status)

            with gr.Tab("Transcript Editing"):
                components = []

                #states
                file_states = gr.State(value=[None]*items_per_page)
                current_page_state = gr.State(value=1)
                page_label_state = gr.State(value="Page 1 of 1")

                #define elements on top
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Data")
                    previous_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                    page_label = gr.Markdown("Page 1 of 1")

                #parts
                for i in range(items_per_page):
                    with gr.Row():
                        audio_component = gr.Audio(visible=False)
                        transcript_box = gr.Textbox(visible=False, label="Transcript", lines=3, interactive=True)
                        save_btn = gr.Button("Update", visible=False)
                        status_box = gr.Textbox(visible=False, label="Status", interactive=False)
                        components.append((audio_component, transcript_box, save_btn, status_box))

                #set outputs to all components, file_states, current_page_state, page_label
                outputs = []
                for component in components:
                    outputs.extend(component)
                outputs.extend([file_states, current_page_state, page_label])

                refresh_btn.click(fn=lambda: refresh_data(1),inputs=[],outputs=outputs) #refresh button

                #prev page
                previous_btn.click(fn=lambda current_page: refresh_data(current_page - 1),inputs=[current_page_state],outputs=outputs)

                #next page
                next_btn.click(fn=lambda current_page: refresh_data(current_page + 1),inputs=[current_page_state],outputs=outputs)

                #update transcript
                for index, (audio_component, transcript_box, save_btn, status_box) in enumerate(components):
                    save_btn.click(save_transcript(index),inputs=[transcript_box, file_states],outputs=status_box)

            with gr.Tab("Post Processing"):
                with gr.Row():
                    san_check = gr.Button("Step 1: Sanity Check")
                    package_data = gr.Button("Step 2: Package Dataset")

                san_status = gr.Textbox(label="Sanity Check Output", interactive=False)
                download_link = gr.File(label="Download Packaged Dataset", interactive=False)

                san_check.click(sanitycheck.run_check, inputs=[], outputs=san_status)
                package_data.click(main_process.zip_output, inputs=[], outputs=[download_link])

            gr.Markdown("Is there an issue? Feel free to open an issue on my [GitHub](https://github.com/DominicTWHV/LJSpeech_Dataset_Generator)")
        return app
    
if __name__ == "__main__":
    ui = LJSpeechDatasetUI(dataset_dir="wavs", metadata_file="metadata.csv")
    app = ui.create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)