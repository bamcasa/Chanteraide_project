import librosa
import librosa.display
import matplotlib.pyplot as plt

def audio_to_waveform_image_librosa(audio_path, image_path):
    # 오디오 파일 로드
    y, sr = librosa.load(audio_path)

    # 웨이브폼 이미지 생성
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 이미지 파일로 저장
    plt.savefig(image_path, dpi=300)

audio_file_path = 'recorded_audio/output.wav'  # 오디오 파일 경로
image_file_path = 'audio_img/audio_waveform_librosa.png'  # 이미지 파일 경로

audio_to_waveform_image_librosa(audio_file_path, image_file_path)
