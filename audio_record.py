import pyaudio
import wave

def record_audio(filename, record_time=5):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    audio = pyaudio.PyAudio()

    print("음성 녹음 시작...")
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    for _ in range(0, int(rate / chunk * record_time)):
        data = stream.read(chunk)
        frames.append(data)

    print("음성 녹음 종료.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

if __name__ == '__main__':
    path = "recorded_audio"
    output_filename = "output.wav"
    record_time = 5  # 녹음 시간(초 단위)
    record_audio(f"{path}/{output_filename}", record_time)
