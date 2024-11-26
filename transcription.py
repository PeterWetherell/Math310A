import ssl
import whisper
import time
# import cleaning
# import check
# import os
from urllib import request

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context


def transcribe(audio_file, textfile):
    model=whisper.load_model("tiny")
    start_time = time.time()
    result = model.transcribe(audio_file)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    #with open('Transcriptions/yellow_wallpaper_transcription.txt','w', encoding="utf-8") as file:
    with open(textfile,'w', encoding="utf-8") as file:
        file.write(result['text'])
    #print(result)
transcribe("./NormalizedSoundData/Clean/YWP.wav", "./Transcriptions/YWP.txt")
# Specify the folder path


# for filename in os.listdir(folder_path):
#     # Construct full file path
#     file_path = os.path.join(folder_path, filename)
#     file_root, file_extension = os.path.splitext(filename)
#     text_path = os.path.join("./OutputTranscriptions", file_root) + '.txt'
#     print("transcribing", filename)
#     transcribe(file_path, text_path)
# Iterate over files in the folder

# folder_path = "./OutputTranscriptions"
# output_folder = "./CleanedOutputTranscriptions"
# my_dict = {}
# for filename in os.listdir(folder_path):
#     # Construct full file path
#     text_path = os.path.join(folder_path, filename)
#     cleaned_path = os.path.join(output_folder,filename)
#     cleaning.process_file(text_path, cleaned_path)
#     model_sim = check.cosine_similarity_files("./Transcriptions/yellow_wallpaper_transcription.txt",cleaned_path)
#     regular_sim = check.cosine_similarity_files("./Transcriptions/yellow_wallpaper_transcription.txt", "./Transcriptions/yellow_wallpaper_cleaned_noisy.txt")
#     my_dict[filename] = model_sim-regular_sim

# # Sort dictionary by values (ascending)
# sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
# for i in sorted_dict:
#     print(i, " ", sorted_dict[i])
