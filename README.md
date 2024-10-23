# Math310A
This is the github repository for the Math 310 A level for the group Fourier Folks. This project focuses on generating noise and noise cancelation.

The project generates .wav files with noise using custom profiles, it then checks to see how distorted the audio file is using whisper (an AI text to speach program) to give a baseline for how it affected its discernability. We can then apply temporal and frequency based filters to do our best to remove the noise and run it back through Wisper to see how effective the denoising is at recovering the origional sound file.
