import ssl
import whisper
import time

from urllib import request

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context


def transcribe(audio_file):
    model=whisper.load_model("tiny")
    start_time = time.time()
    result = model.transcribe(audio_file)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    with open('Transcriptions/pnp_part2_transcription_noisy.txt','w', encoding="utf-8") as file:
        file.write(result['text'])
    #print(result)
transcribe("./NormalizedSoundData/Noisy/PAP2.wav")