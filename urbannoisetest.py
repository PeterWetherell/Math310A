""""
#It would be nice to get this working because it would be audio data we can add ontop of the audio as noise but for now it doesn't

import soundata

# learn wich datasets are available in soundata
print(soundata.list_datasets())

# choose a dataset and download it
dataset = soundata.initialize('urbansound8k', data_home='/choose/where/data/live')
dataset.download()

# get annotations and audio for a random clip
example_clip = dataset.choice_clip()
tags = example_clip.tags
y, sr = example_clip.audio
"""