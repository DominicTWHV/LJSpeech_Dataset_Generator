import os
import io
import wave
import subprocess
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

#check dirs
os.makedirs('wavs', exist_ok=True)
os.makedirs('input', exist_ok=True)

#init sr
recognizer = sr.Recognizer()

def split_audio(input_file, base_filename):
    """Splits audio into 2-11 second chunks, stored in /wavs."""
    print(f"[DEBUG] Loading audio file: {input_file}")
    audio = AudioSegment.from_wav(input_file)
    original_duration = len(audio)

    print(f"[DEBUG] Splitting audio into chunks based on silence...")
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    prepped_audio = []
    temp_chunk = AudioSegment.empty()  #init chunk for splitting
    for i, chunk in enumerate(chunks):
        chunk_duration = len(chunk)
        
        if chunk_duration < 2000:  #accu chunk if too short
            temp_chunk += chunk
        else:
            if len(temp_chunk) > 0:  #proc accumulated audio
                temp_chunk += chunk  #merge
                chunk_duration = len(temp_chunk)
                chunk = temp_chunk
                temp_chunk = AudioSegment.empty()  #reset chunk

            if chunk_duration <= 11000:
                chunk_name = f"{base_filename}-{i:04d}.wav"
                chunk_path = os.path.join('wavs', chunk_name)  #save in the 'wavs' folder
                print(f"[DEBUG] Saving chunk {i}: {chunk_name} (Duration: {chunk_duration / 1000:.2f} seconds)")
                chunk.export(chunk_path, format="wav")
                prepped_audio.append(chunk_path)
            else:
                print(f"[DEBUG] Skipping chunk {i}: Duration {chunk_duration / 1000:.2f} seconds exceeds limit")

    #handle chunks at the end
    if len(temp_chunk) > 0:
        chunk_name = f"{base_filename}-last.wav"
        chunk_path = os.path.join('wavs', chunk_name)
        temp_chunk.export(chunk_path, format="wav")
        prepped_audio.append(chunk_path)
        print(f"[DEBUG] Saving remaining chunk: {chunk_name} (Duration: {len(temp_chunk) / 1000:.2f} seconds)")

    return prepped_audio, original_duration

def transcribe_audio(audio_file):
    """Transcribes audio using SpeechRecognition with Google Speech-to-Text."""
    print(f"[DEBUG] Transcribing audio file: {audio_file}")
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        
        try:
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

    total_original_duration = 0  #to store the total length of the input files
    total_processed_duration = 0  #to store the total length of the processed chunks

    for wav_file in wav_files:
        base_filename = os.path.splitext(wav_file)[0]  #strip file extension
        input_path = os.path.join(input_dir, wav_file)
        print(f"[DEBUG] Processing file: {wav_file}")
        
        #split the audio and generate chunks
        prepped_audio_chunks, original_duration = split_audio(input_path, base_filename)
        total_original_duration += original_duration

        #transcribe each chunk and add to metadata
        for chunk_file in prepped_audio_chunks:
            transcript = transcribe_audio(chunk_file)
            chunk_audio = AudioSegment.from_wav(chunk_file)
            total_processed_duration += len(chunk_audio)  #add chunk duration to processed total
            chunk_name = os.path.basename(chunk_file)  #get the filename for the chunk
            metadata.append([os.path.join('wavs', chunk_name), transcript])  #relative path for CSV
    
    #create the CSV in LJSpeech format
    print(f"[DEBUG] Writing metadata.csv...")
    df = pd.DataFrame(metadata, columns=["wav_filename", "transcript"])
    df.to_csv("metadata.csv", index=False)  # Save the metadata at the root
    print(f"[DEBUG] metadata.csv generated successfully")

    #calculate and print the retained/lost percentage
    retained_percentage = (total_processed_duration / total_original_duration) * 100
    lost_percentage = 100 - retained_percentage
    print(f"[DEBUG] Total original audio duration: {total_original_duration / 1000:.2f} seconds")
    print(f"[DEBUG] Total processed audio duration: {total_processed_duration / 1000:.2f} seconds")
    print(f"[DEBUG] Audio retained: {retained_percentage:.2f}%")
    print(f"[DEBUG] Audio lost: {lost_percentage:.2f}%")

    return retained_percentage

def zip_output(output_filename="dataset.zip"):
    """Zips the wavs directory and metadata.csv into the output zip file."""
    print(f"[DEBUG] Zipping the output files into {output_filename}...")
    try:
        subprocess.run(['zip', '-r', output_filename, 'wavs', 'metadata.csv'], check=True)
        print(f"[DEBUG] Successfully created {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error while zipping: {e}")

def main():
    print(f"[DEBUG] Starting the audio processing pipeline...")
    #process wav files and check the retained percentage
    retained_percentage = process_wav_files('input')

    #ensure at least 90% of audio is retained before zipping
    if retained_percentage >= 90:
        zip_output()
        print(f"[DEBUG] Pipeline finished successfully with {retained_percentage:.2f}% audio retained!")
    else:
        print(f"[ERROR] Audio retention below acceptable threshold: {retained_percentage:.2f}%. Skipping zipping.")

if __name__ == "__main__":
    main()
