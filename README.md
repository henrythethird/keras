# Keras Text Generator

This project creates a machine learning model for arbitrary text generation. 

## Structure

The input, as well as the output of the model is 
[one-hot](https://en.wikipedia.org/wiki/One-hot) encoded with the size of the 
alphabet (`x*`). Since RNNs (specifically LSTMs) respond superbly to 
multisequence training, the input data is structured as `(x*, seq, 1)`. From 
the input data, being a sequence of (one-hot encoded) characters, the next 
character in the sequence is deduced. A sequence of `100` characters was chosen
arbitrarily.


The resulting model is sequential `(x*, 100, 1)-(x*)` and structured as follows:

```
LSTM(256, (x*, 100, 1))
   ||
Dropout(0.4)
   ||
LSTM(256)
   ||
Dropout(0.4)
   ||
LSTM(256)
   ||
Dropout(0.4)
   ||
Dense(x*)
```

## Installation

In order to run the model, a working python environment is needed.
The following packages have to be installed manually:
```
pip install keras numpy
```

## Execution

Once `keras` and `numpy` are installed successfully, the `train.py` file can be
executed in the project folder:

```
python train.py
```

The resulting models are saved (per epoch) into the `models/` directory and can
later be reused to resume training.