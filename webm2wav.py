import os
from pydub import AudioSegment

def convert_webm_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="webm")
    audio.export(output_file, format="wav")

folder_path = ".\Q-samples"  # 실제 P-samples 폴더 경로로 변경해야 합니다.

print(os.listdir(folder_path))

for file_name in os.listdir(folder_path):
    print(file_name)
    if file_name.endswith(".webm"):
        input_file = os.path.join(folder_path, file_name)
        output_file = os.path.join(
            folder_path, file_name.rstrip(".webm") + ".wav"
        )
        convert_webm_to_wav(input_file, output_file)
