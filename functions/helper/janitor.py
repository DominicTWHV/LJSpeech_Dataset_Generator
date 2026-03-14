from pathlib import Path


class Janitor:

    @staticmethod
    def reset_dataset_files():
        root_dir = Path(__file__).resolve().parents[2]
        wavs_dir = root_dir / 'wavs'
        metadata_file = root_dir / 'metadata.csv'
        dataset_file = root_dir / 'output' / 'dataset.zip'

        wavs_deleted = 0
        if wavs_dir.is_dir():
            for entry in wavs_dir.iterdir():
                if entry.is_file():
                    entry.unlink()
                    wavs_deleted += 1

        metadata_deleted = False
        dataset_deleted = False

        if metadata_file.is_file():
            metadata_file.unlink()
            metadata_deleted = True

        if dataset_file.is_file():
            dataset_file.unlink()
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
