import os
from pydub import AudioSegment

folder_path = r"samples"
output_file = r"output_merged.wav"

wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
wav_files.sort()

merged = AudioSegment.empty()
for wav_file in wav_files:
    audio = AudioSegment.from_wav(os.path.join(folder_path, wav_file))
    merged += audio

merged.export(os.path.join(folder_path, output_file), format="wav")
print(f"Merged {len(wav_files)} files into {output_file}")
