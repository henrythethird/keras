import numpy
import keras

def create_model(batch_size, old_model = None):
    model = keras.models.Sequential()
    # Building a sequential model (LSTM -> Dropout -> Dense)
    model.add(keras.layers.LSTM(512, batch_size=batch_size, input_shape=(dataX.shape[1], len(chars)), stateful=True, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(512, stateful=True, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(512, stateful=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(chars), activation='softmax'))

    if (old_model != None):
        model.set_weights(old_model.get_weights())

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model



filename = "poems.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()[0:269701]

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

dataX = []
dataY = []

for i in range(0, len(raw_text) - 1):
    dataX.append(char_to_int[raw_text[i]])
    dataY.append(char_to_int[raw_text[i + 1]])

# Reshaping the vector to a LSTM friendly format
dataX = keras.utils.to_categorical(dataX)
dataX = numpy.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
dataY = keras.utils.to_categorical(dataY)

batch_size = 100
model = create_model(batch_size)

filepath="models/weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

for n in range(1,10):
    print("### Epoch " + str(n))
    model.fit(dataX, dataY, epochs=1, batch_size=batch_size, callbacks=[checkpoint])
    model.reset_states()


model = create_model(1, model)

start = 'p'
text = start
for i in range(500):
    inp = keras.utils.to_categorical([char_to_int[start]], len(chars))
    inp = numpy.reshape(inp, (1, 1, inp.shape[1]))

    prediciton = model.predict(inp, verbose=0, batch_size=1)
    best_prediction = numpy.argmax(prediciton)

    start = int_to_char[best_prediction]
    text += start

print(text)
