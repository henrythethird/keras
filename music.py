import tensorflow as tf
from tensorflow.contrib import rnn
import json

def get_training_data():
    music_raw = open("music.json", "r").read()
    music = json.loads(music_raw)
    all_tracks = music["tracks"]
    music_tracks = [track for track in all_tracks if track["length"] > 0]
    piano_track = music_tracks[0]

    reformatted = {}

    for note in piano_track["notes"]:
        start = round(note["time"] * 10)
        end = round((note["time"] + note["duration"]) * 10)
        name = note["name"]

        if name not in reformatted:
            reformatted[name] = [0 for t in range(0, piano_track["length"] * 10)]

        for time in range(start, end):
            reformatted[name][time] = 1

    return reformatted

data = get_training_data()

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input,1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

vocab_size = len(data)
n_input = 3
# number of units in RNN cell
n_hidden = 512

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}