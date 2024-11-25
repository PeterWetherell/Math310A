import cleaning
import check
import os

folder_path = "./OutputTranscriptions"
output_folder = "./CleanedOutputTranscriptions"
my_dict = {}
for filename in os.listdir(folder_path):
    # Construct full file path
    text_path = os.path.join(folder_path, filename)
    cleaned_path = os.path.join(output_folder,filename)
    cleaning.process_file(text_path, cleaned_path)
    model_sim = check.cosine_similarity_files("./Transcriptions/yellow_wallpaper_transcription.txt",cleaned_path)
    regular_sim = check.cosine_similarity_files("./Transcriptions/yellow_wallpaper_transcription.txt", "./Transcriptions/yellow_wallpaper_cleaned_noisy.txt")
    my_dict[filename] = model_sim-regular_sim

# Sort dictionary by values (ascending)
sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
for i in sorted_dict:
    print(i, " ", sorted_dict[i])