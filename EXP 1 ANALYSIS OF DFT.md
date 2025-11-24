# EXP 1 :  ANALYSIS OF DFT WITH AUDIO SIGNAL

# AIM: 

  To analyze audio signal by removing unwanted frequency. 

# APPARATUS REQUIRED: 
   
   PC installed with SCILAB/Python. 

# PROGRAM: 

// analyze audio signal
```
# ==============================
# AUDIO DFT ANALYSIS IN COLAB
# ==============================

# Step 1: Install required packages
!pip install -q librosa soundfile

# Step 2: Upload audio file
from google.colab import files
uploaded = files.upload()   # choose your .wav / .mp3 / .flac file
filename = next(iter(uploaded.keys()))
print("Uploaded:", filename)

# Step 3: Load audio
import librosa, librosa.display
import numpy as np
import soundfile as sf

y, sr = librosa.load(filename, sr=None, mono=True)  # keep original sample rate
duration = len(y) / sr
print(f"Sample rate = {sr} Hz, duration = {duration:.2f} s, samples = {len(y)}")

# Step 4: Play audio
from IPython.display import Audio, display
display(Audio(y, rate=sr))

# Step 5: Full FFT (DFT) analysis
import matplotlib.pyplot as plt

n_fft = 2**14   # choose large power of 2 for smoother spectrum
Y = np.fft.rfft(y, n=n_fft)
freqs = np.fft.rfftfreq(n_fft, 1/sr)
magnitude = np.abs(Y)

plt.figure(figsize=(12,4))
plt.plot(freqs, magnitude)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum (linear scale)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.semilogy(freqs, magnitude+1e-12)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log scale)")
plt.title("FFT Magnitude Spectrum (log scale)")
plt.grid(True)
plt.show()

# Step 6: Top 10 dominant frequencies
N = 10
idx = np.argsort(magnitude)[-N:][::-1]
print("\nTop 10 Dominant Frequencies:")
for i, k in enumerate(idx):
    print(f"{i+1:2d}. {freqs[k]:8.2f} Hz  (Magnitude = {magnitude[k]:.2e})")

# Step 7: Spectrogram (STFT)
n_fft = 2048
hop_length = n_fft // 4
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.ylim(0, sr/2)
plt.show()
```
# AUDIO SIGNAL USED :

[080450_good-night-38118.mp3](https://github.com/user-attachments/files/22921627/080450_good-night-38118.mp3)

# OUTPUT: 
<img width="1300" height="502" alt="image" src="https://github.com/user-attachments/assets/4ea73e27-353b-4921-958c-4d33b13ce852" />

<img width="1364" height="500" alt="image" src="https://github.com/user-attachments/assets/02e4a723-d73d-48b7-810e-3bea4a2993b5" />

Top 10 Dominant Frequencies:
 1.   251.95 Hz  (Magnitude = 2.28e+01)
 2.   253.42 Hz  (Magnitude = 2.15e+01)
 3.   250.49 Hz  (Magnitude = 2.08e+01)
 4.   276.86 Hz  (Magnitude = 2.05e+01)
 5.   278.32 Hz  (Magnitude = 2.03e+01)
 6.   254.88 Hz  (Magnitude = 1.88e+01)
 7.   246.09 Hz  (Magnitude = 1.86e+01)
 8.   244.63 Hz  (Magnitude = 1.80e+01)
 9.   249.02 Hz  (Magnitude = 1.78e+01)
10.   247.56 Hz  (Magnitude = 1.76e+01)

<img width="1162" height="489" alt="image" src="https://github.com/user-attachments/assets/1f87f136-8036-4889-915a-a7e5cc7ffa19" />

# RESULTS :
Thus,Analysis of DFT with Audio Signal is successfully implemented using Python.

