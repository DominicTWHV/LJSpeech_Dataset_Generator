import os

class SanityChecker:
    def __init__(self, metadata_file='metadata.csv', wav_directory='wavs'):
        self.metadata_file = metadata_file
        self.wav_directory = wav_directory

    def count_entries_in_metadata(self):
        if not os.path.isfile(self.metadata_file):
            return None
        with open(self.metadata_file, 'r') as file:
            return sum(1 for line in file if line.strip())

    def count_wav_files(self):
        if not os.path.isdir(self.wav_directory):
            return None
        return len([f for f in os.listdir(self.wav_directory) if f.lower().endswith('.wav')])

    def run_check(self):
        metadata_entries = self.count_entries_in_metadata()
        if metadata_entries is None:
            return "Critical Error: metadata.csv file not found!"

        actual_wav_count = self.count_wav_files()
        if actual_wav_count is None:
            return "Critical Error: wavs/ directory not found!"

        if metadata_entries == actual_wav_count:
            return f"Sanity check passed:\nNumber of .wav files: {actual_wav_count}\nNumber of metadata rows: {metadata_entries}\n\nPlease proceed to zipping the dataset."
        return f"Critical Error: Expected {metadata_entries} wav files from metadata.csv, found {actual_wav_count}."

if __name__ == '__main__':
    checker = SanityChecker()
    checker.run_check()
