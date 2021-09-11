import librosa
import librosa.display
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split


audio_path = 'Audios/audio/'
metadata = pd.read_csv('Audios/metadata/FanSounds.csv')

def parser(row):
    fn = ((audio_path) + str(row["slice_file_name"]))
    # print(fn)
    x, sr = librosa.load(fn, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
    feature = mfccs
    label = row["class"]
    return [feature, label]

data = metadata.apply(parser, axis=1)
data.columns = ['feature', 'label']
# print(data.columns)
x = np.array(list(zip(*data))[0])
y = np.array(list(zip(*data))[1])
print(x)
print(y)


le = LabelEncoder()
y = np_utils.to_categorical(le.fit_transform(y))
# print(y)


num_classes = 2
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
# print(model.summary())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# model.summary()
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2, class_weight=None)

# test_accuracy = model.evaluate(x_test, y_test, verbose=0)
# print(test_accuracy[1])



from keras.models import load_model
model.save("my_model.h5")



