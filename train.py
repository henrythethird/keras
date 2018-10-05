import numpy
import keras
from batch_generator import BatchGenerator


def load_data(file_name, seq_length, batch_size):
    raw_text = open(file_name).read().lower()
    bc = BatchGenerator(raw_text, seq_length, batch_size)

    return bc

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

def train(model, generator, batch_size, epochs):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='models/weights-{epoch:02d}-{loss:.4f}.hdf5', 
        monitor='loss', 
        verbose=False, 
        save_best_only=True, 
        mode='min'
    )

    model.fit_generator(
        generator.generate(), 
        epochs=epochs, 
        shuffle=False, 
        callbacks=[checkpoint],
        steps_per_epoch=generator.n_samples
    )

def evaluate(model, generator, seed):
    seq_length = len(seed)
    for _ in range(0, 1000):
        inp = generator._one_hot_encode_list(seed[-seq_length:])
        inp = numpy.reshape(inp, (1, seq_length, generator.vocab_size))

        prediciton = model.predict(inp, verbose=False, batch_size=1)
        start = generator._one_hot_decode(prediciton)
        seed += start

    return seed[seq_length:]

generator = load_data(
    "text.txt", seq_length=100, batch_size=100
)

model = create_model((100, generator.vocab_size), generator.vocab_size)

train(model, generator, batch_size=100, epochs=50)
generated_string = evaluate(
    model, generator,
    'it stands to reason that this sketch of ' + 
    'the saint, made upon the model of the ' + 
    'whole species, can be '
)

print(generated_string)
