import os
import subprocess
import pandas as pd
import speech_recognition as sr
import wave
import re

#create necessary  directories
os.makedirs('wavs', exist_ok=True)

#initialize recognizer
recognizer = sr.Recognizer()

def transcribe_audio(audio_file):
    """Transcribes audio using SpeechRecognition with Google Speech-to-Text."""
    print(f"[DEBUG] Transcribing audio file: {audio_file}")
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        
        try:
            #transcribe audio with sr
            transcript = recognizer.recognize_google(audio_data)
            print(f"[DEBUG] Transcript: {transcript}")
        except sr.UnknownValueError:
            print(f"[WARNING] Google Speech Recognition could not understand {audio_file}")
            transcript = ""
        except sr.RequestError as e:
            print(f"[ERROR] Could not request results from Speech Recognition service; {e}\n\nAre you connected to the internet?")
            transcript = ""
    
    return transcript

def process_wav_files(input_dir):
    """Processes all wav files, transcribes, and generates a CSV."""
    print(f"[DEBUG] Processing .wav files in directory: {input_dir}")
    metadata = []
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"[DEBUG] Found {len(wav_files)} .wav files")

    for wav_file in wav_files:
        input_path = os.path.join(input_dir, wav_file)
        print(f"[DEBUG] Processing file: {wav_file}")
        
        #transcribe and append for each file
        transcript = transcribe_audio(input_path)
        metadata.append([os.path.join('wavs', wav_file), transcript])
    
    #regex match and sort filenames
    metadata.sort(key=lambda x: int(re.search(r'processed(\d+)', x[0]).group(1)))

    #create metadata file
    print(f"[DEBUG] Writing metadata.csv...")
    df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
    df.to_csv("metadata.csv", sep='|', index=False)
    print(f"[DEBUG] metadata.csv generated successfully")

def zip_output(output_filename="output/dataset.zip"):
    """Zips the wavs directory and metadata.csv into the output zip file."""
    print(f"[DEBUG] Zipping the output files into {output_filename}...")
    try:
        #zip file
        subprocess.run(['zip', '-r', output_filename, 'wavs', 'metadata.csv'], check=True)
        print(f"[DEBUG] Successfully created {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error while zipping: {e}")

def total_audio_length(directory):
    total_length = 0
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    for wav_file in wav_files:
        file_path = os.path.join(directory, wav_file)
        with wave.open(file_path, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
            total_length += duration
            
    return total_length

def main():
    print(f"[DEBUG] Starting the audio processing pipeline...")
    process_wav_files('wavs')
    zip_output()
    
    print(f"[OK] Dataset Created!")

if __name__ == "__main__":
    main()
