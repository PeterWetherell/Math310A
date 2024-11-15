import ssl
import whisper
from urllib import request

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context


def transcribe(audio_file):
    model=whisper.load_model("tiny")
    result = model.transcribe(audio_file)
    with open('Transcriptions/pnp_part2_transcription_noisy.txt','w', encoding="utf-8") as file:
        file.write(result['text'])
    #print(result)
transcribe("./NormalizedSoundData/Noisy/PAP2.wav")