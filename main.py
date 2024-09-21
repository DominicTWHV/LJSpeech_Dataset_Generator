import os
import io
import wave
import subprocess
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

#create necessary directories
os.makedirs('wavs', exist_ok=True)
os.makedirs('input', exist_ok=True)

#initialize the recognizer for SpeechRecognition
recognizer = sr.Recognizer()

def split_audio(input_file, base_filename):
    """Splits audio into 2-11 second chunks, stored in /wavs."""
    print(f"[DEBUG] Loading audio file: {input_file}")
    audio = AudioSegment.from_wav(input_file)

    print(f"[DEBUG] Splitting audio into chunks based on silence...")
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    prepped_audio = []
    for i, chunk in enumerate(chunks):
        chunk_duration = len(chunk)
        if 2000 <= chunk_duration <= 11000:  # Ensure between 2-11 seconds
            chunk_name = f"{base_filename}-{i:04d}.wav"
            chunk_path = os.path.join('wavs', chunk_name)  # Save in the 'wavs' folder
            print(f"[DEBUG] Saving chunk {i}: {chunk_name} (Duration: {chunk_duration / 1000:.2f} seconds)")
            chunk.export(chunk_path, format="wav")
            prepped_audio.append(chunk_path)
        else:
            #code will try to split audio into smaller chunks between 2-11s
            print(f"[DEBUG] Skipping chunk {i}: Duration {chunk_duration / 1000:.2f} seconds not in 2-11s range")
    
    return prepped_audio

def transcribe_audio(audio_file):
    """Transcribes audio using SpeechRecognition with Google Speech-to-Text."""
    print(f"[DEBUG] Transcribing audio file: {audio_file}")
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        
        try:
            #transcript audio with speech recog
            transcript = recognizer.recognize_google(audio_data)
            print(f"[DEBUG] Transcript: {transcript}")
        except sr.UnknownValueError:
            print(f"[DEBUG] Google Speech Recognition could not understand {audio_file}")
            transcript = ""
        except sr.RequestError as e:
            print(f"[DEBUG] Could not request results from Google Speech Recognition service; {e}")
            transcript = ""
    
    return transcript

def process_wav_files(input_dir):
    """Processes all wav files, splits them, transcribes, and generates a CSV."""
    print(f"[DEBUG] Processing .wav files in directory: {input_dir}")
    metadata = []
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"[DEBUG] Found {len(wav_files)} .wav files")

    for wav_file in wav_files:
        base_filename = os.path.splitext(wav_file)[0]  # Strip file extension
        input_path = os.path.join(input_dir, wav_file)
        print(f"[DEBUG] Processing file: {wav_file}")
        
        #split the audio and generate chunks
        prepped_audio_chunks = split_audio(input_path, base_filename)
        
        #transcribe each chunk and add to metadata
        for chunk_file in prepped_audio_chunks:
            transcript = transcribe_audio(chunk_file)
            chunk_name = os.path.basename(chunk_file)  # Get the filename for the chunk
            metadata.append([os.path.join('wavs', chunk_name), transcript])  # Relative path for CSV
    
    #create the CSV in LJSpeech format
    print(f"[DEBUG] Writing metadata.csv...")
    df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
    df.to_csv("metadata.csv", index=False)  # Save the metadata at the root
    print(f"[DEBUG] metadata.csv generated successfully")

def zip_output(output_filename="dataset.zip"):
    """Zips the wavs directory and metadata.csv into the output zip file."""
    print(f"[DEBUG] Zipping the output files into {output_filename}...")
    try:
        #create a zip file with the 'wavs' directory and 'metadata.csv'
        subprocess.run(['zip', '-r', output_filename, 'wavs', 'metadata.csv'], check=True)
        print(f"[DEBUG] Successfully created {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error while zipping: {e}")

def main():
    print(f"[DEBUG] Starting the audio processing pipeline...")
    #process WAV files
    process_wav_files('input')
    
    #zip the output using native zip
    zip_output()
    print(f"[DEBUG] Pipeline finished successfully!")

if __name__ == "__main__":
    main()
