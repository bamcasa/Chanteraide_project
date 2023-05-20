import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 데이터 전처리 함수
def preprocess_data(audio_files, labels):
    data = []

    for file in audio_files:
        waveform, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr)
        data.append(mfcc)

    max_shape = max([x.shape[1] for x in data])

    # Zero-padding
    data = [np.pad(x, ((0, 0), (0, max_shape - x.shape[1]))) for x in data]
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    labels_encoded = tf.keras.utils.to_categorical(labels_encoded, num_classes=len(np.unique(labels)))

    return data, labels_encoded


# 모델 생성 함수
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


audio_path = ".\\audio"

# 데이터 로드 및 전처리
audio_files = []
labels = []  # 각 오디오 파일의 라벨
for dir in os.listdir(audio_path):
    for f in os.listdir(os.path.join(audio_path, dir)):
        audio_files.append(os.path.join(audio_path, dir,f))
        labels.append(dir)
# audio_files = [ for f in os.listdir(audio_path)]
print(audio_files)
print(labels)


X, y = preprocess_data(audio_files, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 훈련
input_shape = (X_train.shape[1], X_train.shape[2], 1)
num_classes = len(np.unique(labels))
model = create_model(input_shape, num_classes)

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 성능 검증 및 시각화
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
