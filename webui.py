import os
import pandas as pd
import gradio as gr

from functions.filter import NoiseReducer
from functions.split import AudioSplitter
from functions.main import MainProcess
from functions.sanitycheck import SanityChecker


class LJSpeechDatasetUI:
    def __init__(self, dataset_dir="wavs", metadata_file="metadata.csv"):
        self.dataset_dir = dataset_dir
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            df = pd.read_csv(self.metadata_file, sep="|", header=None, names=["filename", "transcript"], dtype=str)
            df["filename"] = df["filename"].astype(str)
            return df
        raise FileNotFoundError(f"Metadata file '{self.metadata_file}' not found.")

    def load_data(self):
        audio_files = [f for f in os.listdir(self.dataset_dir) if f.endswith(".wav")]
        meta_map = {os.path.basename(r["filename"]): r["transcript"] for _, r in self.metadata.iterrows()}
        return [(os.path.join(self.dataset_dir, af), meta_map.get(af, "")) for af in audio_files]

    def update_metadata(self, audio_file, new_transcript):
        base_name = os.path.basename(audio_file)
        mask = self.metadata["filename"].apply(os.path.basename) == base_name
        if mask.any():
            self.metadata.loc[mask, "transcript"] = new_transcript
            self.metadata.to_csv(self.metadata_file, sep="|", index=False, header=False)
            return f"Transcript for {base_name} updated: {new_transcript}"
        return f"Error: {base_name} not found in metadata."

    def create_interface(self):
        data = self.load_data()

        def process_clip(audio_file, _, new_text):
            return self.update_metadata(audio_file, new_text)

        with gr.Blocks() as app:
            gr.Markdown("## LJSpeech Dataset Generator")

            noise_reducer = NoiseReducer()
            splitter = AudioSplitter()
            main_process = MainProcess()
            sanitycheck = SanityChecker()
            
            with gr.Tab("Preprocessing"):
                

                with gr.Row():
                    pp_filter = gr.Button("Step 1: Preprocess - Filter Background Noise")
                    pp_chunk = gr.Button("Step 2: Preprocess - Chunking")
                    pp_main = gr.Button("Step 3: Preprocess - Auto Transcript")

                pp_status = gr.Textbox(label="Preprocess Status", interactive=False)

                pp_filter.click(noise_reducer.gradio_run, inputs=[], outputs=pp_status)
                pp_chunk.click(splitter.gradio_run, inputs=[], outputs=pp_status)
                pp_main.click(main_process.gradio_run, inputs=[], outputs=pp_status)

            with gr.Tab("Transcript Editing"):
                for audio_file, transcript in data:
                    with gr.Row():
                        gr.Audio(audio_file, label=os.path.basename(audio_file), interactive=False)
                        transcript_box = gr.Textbox(value=transcript, label="Transcript", lines=3)
                        save_btn = gr.Button("Save")
                        save_btn.click(process_clip, [gr.State(audio_file), transcript_box, transcript_box], outputs=gr.Textbox(show_label=False))

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