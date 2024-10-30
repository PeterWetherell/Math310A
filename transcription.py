import ssl
import whisper
from urllib import request

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

model=whisper.load_model("tiny")

def transcribe(audio_file):
    result = model.transcribe(audio_file)
    with open('result.txt','w') as file:
        file.write(result['text'])
    print(result)

transcribe("output.wav")