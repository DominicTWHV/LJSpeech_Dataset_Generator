import os
import re
import tempfile
from pydub import AudioSegment
import speech_recognition as sr

#directories
input_dir = 'wavs/'
output_dir = 'wavs/'

#define max chunk duration, generally you want something between 2000-11000ms
max_chunk_duration = 5000

#define file name pattern to match later on
processed_pattern = re.compile(r'^(.*)_processed(\d+)\.wav$')

def split_audio(filename):
    audio = AudioSegment.from_wav(filename)
    recognizer = sr.Recognizer()

    #total audio duration
    total_duration = len(audio)

    #split
    chunks = []
    for start in range(0, total_duration, max_chunk_duration):
        end = min(start + max_chunk_duration, total_duration)
        chunk = audio[start:end]
        print(f"[DEBUG] Working on tempfile {temp_file.name}")
        #save the chunk to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            chunk.export(temp_file.name, format="wav")
            temp_file.seek(0)  #go back to the beginning of the file

            #recognize the chunk to ensure words don't get cut
            with sr.AudioFile(temp_file.name) as source:
                audio_data = recognizer.record(source)
                try:
                    print(f"[DEBUG] Performing STT on {temp_file.name}")
                    recognizer.recognize_google(audio_data)
                    print(f"[DEBUG] STT finished on {temp_file.name}")
                    chunks.append((start, end))
                except sr.UnknownValueError:
                    continue

    #export
    for i, (start, end) in enumerate(chunks):
        chunk = audio[start:end]
        chunk_filename = os.path.join(output_dir, f"{os.path.basename(filename).split('.')[0]}_processed{i + 1}.wav")
        chunk.export(chunk_filename, format="wav")

#for each file
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    
    if filename.endswith('.wav'):
        if not processed_pattern.match(filename):
            split_audio(file_path)  #process
        else:
            print(f"[DEBUG] Skipping processed file: {filename}")

#delete og files
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    if os.path.isfile(file_path) and not processed_pattern.match(filename):
        os.remove(file_path)
        print(f"[OK] Deleted unprocessed file: {filename}")

print("[OK] Processing complete. Input files have been deleted.")
