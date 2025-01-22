import os

class Janitor:

    @staticmethod
    def reset_dataset_files():
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        wavs_dir = os.path.join(root_dir, 'wavs')
        metadata_file = os.path.join(root_dir, 'metadata.csv')

        wavs_deleted = 0
        if os.path.exists(wavs_dir):
            for filename in os.listdir(wavs_dir):
                file_path = os.path.join(wavs_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    wavs_deleted += 1

        metadata_deleted = False
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            metadata_deleted = True
        
        return f"Deleted {wavs_deleted} files from 'wavs' directory and metadata file: {'deleted' if metadata_deleted else 'not found'}.\n Target Directories:\n{wavs_dir}\n{metadata_file}"