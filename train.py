import numpy
import keras

def load_data(file_name, seq_length):
    raw_text = open(file_name).read().lower()[0:20000]
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

    return dataX, dataY, chars, char_to_int, int_to_char

def create_model(input_shape, output_shape):
    model = keras.models.Sequential()

    # LSTM layers are used, because they are incredibly good at learning
    # sequences of structured data
    model.add(keras.layers.LSTM(
        units=128, 
        input_shape=input_shape,
        return_sequences=True
    ))

    # The dropout layers exist to decrease the overfitting effect
    # Overfitting occurs when the network just learns the text instead of 
    # generalizing
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.LSTM(128, stateful=False, return_sequences=True))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.LSTM(128, stateful=False))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(output_shape, activation='softmax'))

    # Compile the model with some hard coded settings on loss function and 
    # optimizer
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['categorical_accuracy']
    )

    return model

def train(model, dataX, dataY, batch_size, epochs):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='models/weights-{epoch:02d}-{loss:.4f}.hdf5', 
        monitor='loss', 
        verbose=False, 
        save_best_only=True, 
        mode='min'
    )

    model.fit(
        x=dataX, y=dataY, 
        epochs=epochs, 
        batch_size=batch_size, 
        shuffle=False, 
        callbacks=[checkpoint]
    )

def evaluate(model, seed):
    seq_length = len(seed)
    for _ in range(0, 1000):
        inp = [
            keras.utils.to_categorical(char_to_int[v], len(char_to_int)) 
            for v in seed[-seq_length:]
        ]
        inp = numpy.reshape(inp, (1, seq_length, len(char_to_int)))

        prediciton = model.predict(inp, verbose=False, batch_size=1)
        best_prediction = numpy.argmax(prediciton)

        start = int_to_char[best_prediction]
        seed += start

    return seed[seq_length:]

dataX, dataY, chars, char_to_int, int_to_char = load_data(
    "text.txt", seq_length=100
)

model = create_model((dataX.shape[1], dataX.shape[2]), len(chars))

train(model, dataX, dataY, batch_size=100, epochs=50)
generated_string = evaluate(model, 'it stands to reason that this sketch of ' + 
                                   'the saint, made upon the model of the ' + 
                                   'whole species, can be ')

print(generated_string)
