import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load one mic track (change filename if needed)
filename = r"D:\software project\Software-project\recordings\track_1\mic_1.mp3"
 # or .wav, etc.
y, sr = librosa.load(filename)

# Extract pitch (f₀) using pYIN
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C6')
)

# Generate time axis
times = librosa.times_like(f0)

# Plot the waveform and f₀ contour
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.plot(times, f0, color='r', label='f₀ (pitch in Hz)')
plt.title(f"Pitch Contour (f₀) - {filename}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.tight_layout()
plt.show()