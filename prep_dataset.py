import os
import whisper
from pydub import AudioSegment, silence
from datetime import datetime
from tqdm import tqdm

# Initialize the whisper model
#whisper_model = "base"
whisper_model = "large-v2"
model = whisper.load_model(whisper_model, 'cuda')
input = "input.wav"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Check if file exists
if not os.path.exists(input):
    print(f"'{input}' directory does not exist.")
    exit()

def sort_file_by_line_length(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Sort the lines by their length
    lines.sort(key=len)

    with open("sorted_"+filename, 'w', encoding='utf-8') as file:
        file.writelines(lines)

def transcribe(file_paths):
    global timestamp
    
    # Append the timestamp to the train.txt filename
    train_filename = f"train_{timestamp}.txt"
    
    # Open train.txt for writing
    with open(train_filename, "w", encoding='utf-8') as train_file:
        for file_path in tqdm(file_paths, desc="Transcribing"):
            # Extract folder name and file name
            folder_name = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)

            # Transcribe the audio file
            result = model.transcribe(file_path)
            transcript = result['text']

            # Write the result in the desired format
            train_file.write(f"outputs/{folder_name}/{file_name}|{transcript}\n")

    print("Transcription complete!")

def segment_audio(input_file, silence_thresh=-40, min_silence_len=400, min_chunk_len=4000, max_chunk_len=8000, padding=500):
    
    #---------------------------------------------------------------------------------
    # silence_thresh = the silence threshold in dBFS. The default is -40 dBFS.
    # min_silence_len = the minimum length of silence in milliseconds to be considered as a segment boundary.
    # min_chunk_len = the minimum length of a chunk in milliseconds.
    # max_chunk_len = the maximum length of a chunk in milliseconds.
    # padding = silence added to the beginning and end of the audio segments in millisec
    #---------------------------------------------------------------------------------
    global timestamp
    
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Detect non-silent chunks
    chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Create the 'outputs' folder if it doesn't exist
    base_output_folder = "outputs"
    os.makedirs(base_output_folder, exist_ok=True)

    # Create a subfolder with timestamp inside 'outputs'
    output_folder = os.path.join(base_output_folder, f"output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine the padding width for filenames
    num_chunks = len(chunks)
    padding_width = len(str(num_chunks))

    # Initialize an empty list to store file paths
    file_paths = []

    # Filter and export chunks
    os.makedirs(output_folder, exist_ok=True)
    for i, chunk in enumerate(tqdm(chunks, desc="Segmenting"), start=1): # Start numbering from 1
        chunk_length = len(chunk)
        if min_chunk_len <= chunk_length <= max_chunk_len:
            # Add padding
            padded_chunk = AudioSegment.silent(duration=padding) + chunk + AudioSegment.silent(duration=padding)
            
            # Create filename with padded index
            filename = f"chunk_{str(i).zfill(padding_width)}.wav"
            file_path = os.path.join(output_folder, filename)
            padded_chunk.export(os.path.join(output_folder, filename), format="wav")
            
            # Add the file path to the list
            file_paths.append(file_path)
    
    return file_paths

# Segment audio file
file_paths = segment_audio(input)

# Transcribe files
transcribe(file_paths)

# Create a sorted version of the train text file
sort_file_by_line_length(f"train_{timestamp}.txt")
