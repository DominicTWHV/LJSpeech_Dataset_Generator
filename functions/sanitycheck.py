import os

class SanityChecker:
    def __init__(self, metadata_file='metadata.csv', wav_directory='wavs'):
        self.metadata_file = metadata_file
        self.wav_directory = wav_directory

    def count_lines_in_csv(self):
        if not os.path.isfile(self.metadata_file):
            print("\033[91mError: metadata.csv file not found!\033[0m")
            print("Please report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues")
            return "Error: metadata.csv file not found!\nPlease report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues"
        with open(self.metadata_file, 'r') as file:
            return sum(1 for _ in file)

    def count_wav_files(self):
        if not os.path.isdir(self.wav_directory):
            print("\033[91mError: wavs/ directory not found!\033[0m")
            print("Please report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues")
            return "Error: wavs/ directory not found!\nPlease report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues"
        return len([f for f in os.listdir(self.wav_directory) if f.endswith('.wav')])

    def run_check(self):
        num_lines = self.count_lines_in_csv()
        if num_lines is None:
            return "Critical Error: metadata.csv file not found!"
        expected_wav_count = num_lines - 1
        actual_wav_count = self.count_wav_files()
        if actual_wav_count is None:
            return "Critical Error: no .wav files found!"
        if expected_wav_count == actual_wav_count:
            return f"Sanity check passed:\nNumber of .wav files: {actual_wav_count}\nNumber of lines in metadata.csv (excluding header): {num_lines-1}\n\nPlease proceed to zipping the dataset."
        else:
            print("\033[91mCritical Error: Expected {}, found {}.\033[0m".format(expected_wav_count, actual_wav_count))
            print("Please report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues")

if __name__ == '__main__':
    checker = SanityChecker()
    checker.run_check()
