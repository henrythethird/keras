import numpy
import keras

def create_model(batch_size, old_model = None):
    model = keras.models.Sequential()
    # Building a sequential model (LSTM -> Dropout -> Dense)
    model.add(keras.layers.LSTM(256, batch_size=batch_size, input_shape=(dataX.shape[1], dataX.shape[2]), return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(256, stateful=False, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(256, stateful=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(len(chars), activation='softmax'))

    if (old_model != None):
        model.set_weights(old_model.get_weights())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model


filename = "poems.txt"
seq_length = 100
raw_text = open(filename).read().lower()[0:20000]
input_length = len(raw_text)

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

dataX = []
dataY = []

for i in range(0, input_length - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]

    translated_in = keras.utils.to_categorical([char_to_int[c] for c in seq_in], vocab_size)
    translated_out = keras.utils.to_categorical(char_to_int[seq_out], vocab_size)

    dataX.append([translated_in])
    dataY.append([translated_out])

n_patterns = len(dataX)

dataX = numpy.reshape(dataX, (n_patterns, seq_length, vocab_size))
dataY = numpy.reshape(dataY, (n_patterns, vocab_size))

batch_size = 100

model = create_model(batch_size)

filepath="models/weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

model.fit(dataX, dataY, epochs=100, batch_size=batch_size, shuffle=False, callbacks=[checkpoint])

seed = 'it stands to reason that this sketch of the saint, made upon the model of the whole species, can be '
model = create_model(1, model)

for i in range(0, 1000):
    inp = [keras.utils.to_categorical(char_to_int[v], vocab_size) for v in seed[-seq_length:]]
    inp = numpy.reshape(inp, (1, seq_length, vocab_size))

    prediciton = model.predict(inp, verbose=0, batch_size=1)
    best_prediction = numpy.argmax(prediciton)

    start = int_to_char[best_prediction]
    seed += start

print(seed[seq_length:])
