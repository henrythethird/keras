import numpy
import keras

class BatchGenerator:
    def __init__(self, data, sequence_length, batch_size):
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.data_size = len(data)
        self.character_set = sorted(list(set(data)))
        self.vocab_size = len(self.character_set)
        self.n_samples = self.data_size // self.batch_size

        self.char_to_int = dict((c, i) for i, c in enumerate(self.character_set))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.character_set))

    def _one_hot_encode(self, seq):
        return keras.utils.to_categorical(
            self._encode(seq),
            self.vocab_size
        )
    
    def _one_hot_encode_list(self, l):
        return [self._one_hot_encode(el) for el in l]

    def _one_hot_decode(self, enc):
        index = numpy.argmax(enc)
        return self._decode(index)

    def _encode(self, char_var):
        return self.char_to_int[char_var]

    def _decode(self, int_var):
        return self.int_to_char[int_var]

    def generate(self):
        for start in range(self.data_size - self.sequence_length):
            dataX = []
            dataY = []

            for _ in range(self.batch_size):
                end = start + self.sequence_length

                sequence = self._one_hot_encode_list(self.data[start:end])
                sequence = numpy.array(sequence)
                dataX.append(sequence)

                target = self._one_hot_encode(self.data[end])
                dataY.append(target)
            
            dataX = numpy.reshape(
                dataX,
                (self.batch_size, self.sequence_length, self.vocab_size)
            )

            dataY = numpy.reshape(
                dataY,
                (self.batch_size, self.vocab_size)
            )

            yield dataX, dataY
