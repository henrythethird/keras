import numpy
import keras

def try_model(old):
    model = create_model(1, old)

    start = 'a'
    text = start
    for i in range(100):
        inp = keras.utils.to_categorical([char_to_int[start]], len(chars))
        inp = numpy.reshape(inp, (1, 1, inp.shape[1]))

        prediciton = model.predict(inp, verbose=0, batch_size=1)
        best_prediction = numpy.argmax(prediciton)

        start = int_to_char[best_prediction]
        text += start

    print(text)

def create_model(batch_size, old_model = None):
    model = keras.models.Sequential()
    # Building a sequential model (LSTM -> Dropout -> Dense)
    model.add(keras.layers.LSTM(128, batch_size=batch_size, input_shape=(dataX.shape[1], len(chars)), stateful=(old_model == None)))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(keras.layers.LSTM(128, stateful=True))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(keras.layers.LSTM(512, stateful=True))
    #model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.Dense(len(chars), activation='softmax'))

    if (old_model != None):
        model.set_weights(old_model.get_weights())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model



filename = "poems.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

dataX = [char_to_int[i] for i in raw_text[0:500000]]
dataY = [char_to_int[i] for i in raw_text[1:500001]]

testX = [char_to_int[i] for i in raw_text[500000:600000]]
testY = [char_to_int[i] for i in raw_text[500001:600001]]

seq_length = 100

# Reshaping the vector to a LSTM friendly format
dataX = keras.utils.to_categorical(dataX, len(chars))
dataX = numpy.reshape(dataX, (dataX.shape[0], seq_length, dataX.shape[1]))
dataY = keras.utils.to_categorical(dataY, len(chars))

testX = keras.utils.to_categorical(testX, len(chars))
testX = numpy.reshape(testX, (testX.shape[0], seq_length, testX.shape[1]))
testY = keras.utils.to_categorical(testY, len(chars))

batch_size = 100
model = create_model(batch_size)

#filepath="models/weights-{epoch:02d}-{loss:.4f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

for n in range(1,  2):
    print("### Epoch " + str(n))
    model.fit(dataX, dataY, epochs=40, batch_size=batch_size, shuffle=False)
    model.reset_states()
    scores = model.evaluate(dataX, dataY, batch_size=batch_size, verbose=False)
    print('Training data: loss=%f, accuracy=%f' % (scores[0], scores[1]))
    scores = model.evaluate(testX, testY, batch_size=batch_size, verbose=False)
    print('Test data: loss=%f, accuracy=%f' % (scores[0], scores[1]))
    model.reset_states()
    try_model(model)
    model.reset_states()

