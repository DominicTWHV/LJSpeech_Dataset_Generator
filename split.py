import os
import re
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr

#directories
input_dir = 'wavs/'
output_dir = 'wavs/'

#define max chunk duration, generally you want something between 2000-11000ms, 5000ms would be a good balance generally
max_chunk_duration = 5000

#define file name pattern to match later on
processed_pattern = re.compile(r'^(.*)_processed(\d+)\.wav$')

def split_audio(filename):
    audio = AudioSegment.from_wav(filename)
    recognizer = sr.Recognizer()

    #total audio duration
    total_duration = len(audio)
    #split
    raw_chunks = split_on_silence(
        audio,
        min_silence_len=300,
        silence_thresh=audio.dBFS - 16,
        keep_silence=150
    )

    chunks = []
    current_pos = 0
    for raw_chunk in raw_chunks:
        #save the chunk to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            print(f"[DEBUG] Working on tempfile {temp_file.name}")
            raw_chunk.export(temp_file.name, format="wav")
            temp_file.seek(0)  #go back to the beginning of the file

            #recognize the chunk to ensure words don't get cut
            with sr.AudioFile(temp_file.name) as source:
                audio_data = recognizer.record(source)
                try:
                    print(f"[DEBUG] Performing STT on {temp_file.name}")
                    recognizer.recognize_google(audio_data)
                    print(f"[DEBUG] STT finished on {temp_file.name}")

                    chunk_length = len(raw_chunk)
                    end_pos = current_pos + chunk_length
                    if chunk_length > max_chunk_duration:
                        # further manual chunking if needed
                        sub_start = 0
                        while sub_start < chunk_length:
                            sub_end = min(sub_start + max_chunk_duration, chunk_length)
                            chunks.append((current_pos + sub_start, current_pos + sub_end))
                            sub_start += max_chunk_duration
                        current_pos = end_pos
                    else:
                        chunks.append((current_pos, end_pos))
                        current_pos = end_pos
                except sr.UnknownValueError:
                    pass
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
