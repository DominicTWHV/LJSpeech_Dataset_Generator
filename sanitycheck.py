import os

def count_lines_in_csv(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

def count_wav_files(directory):
    return len([f for f in os.listdir(directory) if f.endswith('.wav')])

def main():
    #csv path
    metadata_file = 'metadata.csv'
    
    #check metadata actually exist
    if not os.path.isfile(metadata_file):
        print("\033[91mError: metadata.csv file not found!\033[0m")
        print("Please report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues")
        return

    #count lines
    num_lines = count_lines_in_csv(metadata_file)
    
    #exclude format header
    expected_wav_count = num_lines - 1
    
    #wavs dir
    wav_directory = 'wavs'
    
    #check dir exists
    if not os.path.isdir(wav_directory):
        print("\033[91mError: wavs/ directory not found!\033[0m")
        print("Please report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues")
        return
    
    #count num of wav files
    actual_wav_count = count_wav_files(wav_directory)
    
    #check if match
    if expected_wav_count == actual_wav_count:
        print("\033[92mSanity check success: The number of .wav files matches the expected count.\033[0m")
    else:
        print("\033[91mCritical Error: Mismatch! Expected {}, but found {} .wav files.\033[0m".format(expected_wav_count, actual_wav_count))
        print("Please report this issue at: https://github.com/DominicTWHV/LJSpeech_Dataset_Generator/issues")

if __name__ == '__main__':
    main()
