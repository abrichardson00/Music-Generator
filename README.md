# Music-Generator
Generates a sequence of midi chords using a neural network trained on a dataset of chorales by J.S Bach.
Chorale dataset downloaded from https://archive.ics.uci.edu/ml/datasets/Bach+Choral+Harmony.

Given 4 adjacent chords, the network should hopefully predict an appropriate next chord.
One can repeatedly predict new chords to generate a piece of music, such as https://soundcloud.com/user-542699384/generated-music-1.

One chord is a 1x24 array, eg [0,0,0,1,...,1,0.0.1].
The input for the network is 4 chords concatenated together, i.e. a 1x96 array.
The network is trained on this input, and a desired output which is just the next chord in a chorale.

This script makes use of NN.py (and Neuron.py), from: https://github.com/abrichardson00/Neural-Network.
