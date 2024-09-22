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

#initialize the recognizer for sr
recognizer = sr.Recognizer()

def split_audio(input_file, base_filename):
    #Audio splitting
    
    print(f"[DEBUG] Loading audio file: {input_file}")
    audio = AudioSegment.from_wav(input_file)

    print(f"[DEBUG] Transcribing audio to detect word boundaries...")
    with sr.AudioFile(input_file) as source:
        audio_data = recognizer.record(source)
        try:
            #sr with timestamps
            response = recognizer.recognize_google(audio_data, show_all=True)
            if not response or 'alternative' not in response:
                print(f"[DEBUG] No transcription results for {input_file}")
                return []

            #extract timings
            words_info = response['alternative'][0].get('words', [])
            if not words_info:
                print(f"[DEBUG] No word timings found for {input_file}")
                return []

            prepped_audio = []
            for i, word_info in enumerate(words_info):
                start_time = word_info['startTime']  
                end_time = word_info['endTime']      
                chunk_duration = end_time - start_time

                #convert times from 'HH:MM:SS.milliseconds' to milliseconds
                start_ms = int(float(start_time.split('s')[0]) * 1000)
                end_ms = int(float(end_time.split('s')[0]) * 1000)

                if 2000 <= chunk_duration * 1000 <= 11000:  #ensure between 2-11s
                    chunk_name = f"{base_filename}-{i:04d}.wav"
                    chunk_path = os.path.join('wavs', chunk_name)
                    print(f"[DEBUG] Saving chunk {i}: {chunk_name} (Duration: {chunk_duration:.2f} seconds)")
                    chunk = audio[start_ms:end_ms]
                    chunk.export(chunk_path, format="wav")
                    prepped_audio.append(chunk_path)
                else:
                    print(f"[DEBUG] Skipping chunk {i}: Duration {chunk_duration:.2f} seconds not in 2-11s range")

            return prepped_audio

        except sr.UnknownValueError:
            print(f"[DEBUG] Google Speech Recognition could not understand {input_file}")
            return []
        except sr.RequestError as e:
            print(f"[DEBUG] Could not request results from Google Speech Recognition service; {e}")
            return []


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
        
def total_audio_length(directory):

    total_length = 0
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    for wav_file in wav_files:
        file_path = os.path.join(directory, wav_file)
        with wave.open(file_path, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
            total_length += duration
            
    return total_length

def calculate_audio_retention():

    original_length = total_audio_length('input')
    processed_length = total_audio_length('wavs')
    
    retained_percentage = (processed_length / original_length) * 100 if original_length > 0 else 0
    lost_percentage = 100 - retained_percentage
    
    print(f"[DEBUG] Original Total Length: {original_length:.2f} seconds")
    print(f"[DEBUG] Processed Total Length: {processed_length:.2f} seconds")
    print(f"[DEBUG] Retained Percentage: {retained_percentage:.2f}%")
    print(f"[DEBUG] Lost Percentage: {lost_percentage:.2f}%")
    
def main():
    
    print(f"[DEBUG] Starting the audio processing pipeline...")
    process_wav_files('input')
    zip_output()
    
    # Calculate audio retention statistics
    calculate_audio_retention()
    
    print(f"[DEBUG] Pipeline finished successfully!")
    
if __name__ == "__main__":
    main()
