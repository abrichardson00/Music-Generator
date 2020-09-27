from NN import NN
from  music_data_handler import MusicHandler
import numpy as np
import random

'''
A script for training a neural network on chorales by J.S Bach, downloaded from
https://archive.ics.uci.edu/ml/datasets/Bach+Choral+Harmony
Given 4 adjacent chords, the network should hopefully predict an appropriate next chord.
One can repeatedly predict new chords to generate a piece of music.

One chord is a 1x24 array, eg [0,0,0,1,...,1,0.0.1]
The input for the network is 4 chords concatenated together, i.e. a 1x96 array

This script makes use of NN.py (and Neuron.py), from:
'''


def getTrainingSet(choraleData):
    trainingSet = []
    for i in range(len(choraleData)-4):
        input = []
        for j in range(4):
            for k in range(24):
                input.append(choraleData[i + j][k])

        output = choraleData[i + 4]
        trainingSet.append((input,output))
    return trainingSet


### initialise a network
network = NN(96,[80,60,40,24])

### train the network ------------------------------------------------------
batchSize = 20
for i in range(20):
    ### get the training data
    print("Training iteration: " + str(i))
    trainingSet = []
    ### each training iteration trains the network on 3 chorales at a time
    for j in range(3):
        randomChoraleNo = i*3 + j
        choraleArray = MusicHandler.getChoraleArray(i,'jsbach_chorals_harmony.data')
        trainingSet.extend(getTrainingSet(choraleArray))
    print("Length of training set: " + str(len(trainingSet)))

    ### crop training data to a multiple of batchSize
    trainingDataLength = (len(trainingSet) // batchSize)*batchSize
    print(trainingDataLength)
    trainingData = trainingSet[:trainingDataLength]
    print("Length of cropped training data: " + str(len(trainingData)))
    ### do the network training
    network.SGD(trainingData,100,batchSize,1,None)


network.saveToFile('BachGenerator1')

### generate a piece of music ----------------------------------------------
choraleArray = MusicHandler.getChoraleArray(0,'jsbach_chorals_harmony.data')
choraleLength = 40
chords = []
chords.append(choraleArray[0])
chords.append(choraleArray[1])
chords.append(choraleArray[2])
chords.append(choraleArray[3])
for chord in range(choraleLength-4):
    input = [note for note in chords[chord]]
    for i in range(1,4):
        input = input + chords[chord+i]
    nextPredictedChord = [ int(x) for x in np.rint(network.evaluateInput(input)).tolist()]
    chords.append(nextPredictedChord)

MusicHandler.arrayToMidi(chords,'generation1')
