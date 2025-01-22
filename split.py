import os
import re
import math
import tempfile
from pydub import AudioSegment
import speech_recognition as sr

# Directories
input_dir = 'wavs/'
output_dir = 'wavs/'

# Desired chunk duration (ms) and minimum acceptable chunk duration (ms)
max_chunk_duration = 8000
min_chunk_duration = 4000

# Pattern for processed files
processed_pattern = re.compile(r'^(.*)_processed(\d+)\.wav$')

def split_audio(filename):
    audio = AudioSegment.from_wav(filename)
    recognizer = sr.Recognizer()

    total_duration = len(audio)
    # Calculate number of chunks if each were max_chunk_duration
    num_full_chunks = total_duration // max_chunk_duration
    leftover = total_duration % max_chunk_duration

    # If leftover chunk is too short, merge it with the last chunk
    if leftover < min_chunk_duration and num_full_chunks > 0:
        num_chunks = num_full_chunks
    else:
        num_chunks = num_full_chunks + 1 if leftover else num_full_chunks

    # Approximate chunk length (ms)
    chunk_length = math.ceil(total_duration / num_chunks)

    current_pos = 0
    for i in range(num_chunks):
        end_pos = min(current_pos + chunk_length, total_duration)
        chunk_audio = audio[current_pos:end_pos]

        # Optional: run STT to avoid splitting mid-word (example: skip if too short)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            chunk_audio.export(temp_file.name, format="wav")
            with sr.AudioFile(temp_file.name) as source:
                audio_data = recognizer.record(source)
                try:
                    recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    pass  # If unrecognizable, proceed with the chunk

        out_name = f"{os.path.basename(filename).split('.')[0]}_processed{i+1}.wav"
        chunk_audio.export(os.path.join(output_dir, out_name), format="wav")
        current_pos = end_pos
        if current_pos >= total_duration:
            break

for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    if filename.endswith('.wav') and not processed_pattern.match(filename):
        split_audio(file_path)

# Delete original files
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    if os.path.isfile(file_path) and not processed_pattern.match(filename):
        os.remove(file_path)
