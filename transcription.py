import ssl
import whisper
from urllib import request

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context


def transcribe(audio_file):
    model=whisper.load_model("small")
    result = model.transcribe(audio_file)
    with open('Transcriptions/pnp_part1_transcription.txt','w') as file:
        file.write(result['text'])
    #print(result)
transcribe("Audiobooks/PrideAndPrejudicePart1.wav")