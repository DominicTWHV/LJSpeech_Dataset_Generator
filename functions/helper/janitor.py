import os

class Janitor:

    @staticmethod
    def reset_dataset_files():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        wavs_dir = os.path.join(root_dir, 'wavs')
        metadata_file = os.path.join(root_dir, 'metadata.csv')

        dataset_file = os.path.join(root_dir, 'output/dataset.zip')

        wavs_deleted = 0
        if os.path.exists(wavs_dir):
            for filename in os.listdir(wavs_dir):
                file_path = os.path.join(wavs_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    wavs_deleted += 1

        metadata_deleted = False
        dataset_deleted = False

        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            metadata_deleted = True

        if os.path.exists(dataset_file):
            os.remove(dataset_file)
            dataset_deleted = True
        
        return (
            f"Deleted {wavs_deleted} files from 'wavs' directory.\n"
            f"Metadata file: {'deleted' if metadata_deleted else 'not found'}.\n"
            f"Dataset file: {'deleted' if dataset_deleted else 'not found'}.\n\n"
            f"Target Directories:\n"
            f"  Wavs Directory: {wavs_dir}\n"
            f"  Metadata File: {metadata_file}\n"
            f"  Dataset File: {dataset_file}"
        )
