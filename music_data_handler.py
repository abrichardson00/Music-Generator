# bach music data handler
from mido import Message, MidiFile, MidiTrack




class MusicHandler(object):

    """
    Class for handling data from 'jsbach_chorals_harmony.data', a dataset of 60 chorales by J.S.Bach, downloaded from
    https://archive.ics.uci.edu/ml/datasets/Bach+Choral+Harmony
    Class contains functions for converting data on individual chorales to a python list and converting data to midi etc.
    """

    NoteDict = {
     "C" : 0, "C#" : 1, "Db" : 1, "D" : 2, "D#" : 3, "Eb" : 3, "E" : 4, "E#" : 5, "F" : 5, "F#" : 6, "Gb" : 6, "G" : 7, "G#" : 8, "Ab" : 8, "A" : 9, "A#" : 10, "Bb" : 10, "B" : 11, "B#" : 0
    }

    @staticmethod
    def getChoraleArray(choraleNumber,filename):
        ### read file data
        filein = open(filename, "r")
        i = 0
        #for line in filein.readlines():
        while True:
            line = filein.readline()
            if not line:
                return None
            else:
                tokens = line.split(",")
                if tokens[1] == "1": # we have the start of a chorale
                    if choraleNumber == i: # we have the start of the specified chorale
                        # create list -----------------------------------------------------------
                        choraleData = []

                        choraleID = tokens[0]
                        noteStart = 0
                        while tokens[0] == choraleID:
                            noteLength = int(tokens[15])
                            chordUpper = [1 if t=="YES" else 0 for t in tokens[2:14]]
                            chordBass = [0,0,0,0,0,0,0,0,0,0,0,0]
                            chordBass[MusicHandler.NoteDict[tokens[14]]] = 1
                            chord = chordBass + chordUpper
                            choraleData.append(chord)

                            line = filein.readline()
                            tokens = line.split(",")

                        return choraleData
                        # ----------------------------------------------------------------------------
                    else:
                        i += 1


    # function when given an int i (0 and upwards), parses the file giving the i-th chorale as a midi file
    @staticmethod
    def getChoraleMidi(choraleNumber,filename):
        ### read file data
        filein = open(filename, "r")
        i = 0
        #for line in filein.readlines():
        while True:
            line = filein.readline()
            if not line:
                break
            else:
                tokens = line.split(",")
                if tokens[1] == "1": # we have the start of a chorale
                    if choraleNumber == i: # we have the start of the specified chorale
                        # create midi file -----------------------------------------------------------
                        mid = MidiFile()
                        #for t in range(12):
                        track = MidiTrack()
                        mid.tracks.append(track)

                        choraleID = tokens[0]
                        noteStart = 0
                        while tokens[0] == choraleID:
                            noteLength = int(tokens[15])
                            MusicHandler.ParseLineAsMidi(mid, tokens, noteLength)

                            #noteStart += noteLength
                            line = filein.readline()
                            tokens = line.split(",")

                        mid.save(choraleID + '.mid')
                        # ----------------------------------------------------------------------------
                        break
                    else:
                        i += 1

    @staticmethod
    def ParseLineAsMidi(mid, tokens, noteLength):
        bassVel = 30
        vel = 40
        bassNoteValue = (MusicHandler.NoteDict[tokens[14]] + 36)

        ### notes on -----------------------
        noteValue = 60
        for noteBool in tokens[2:14]:
            if noteBool == "YES":
                mid.tracks[0].append(Message('note_on', note=noteValue, velocity=vel, time=0))
            noteValue+=1

        mid.tracks[0].append(Message('note_on', note=bassNoteValue, velocity=bassVel, time=0)) # bass note on

        ### notes off ----------------------
        noteValue = 60
        noteLen = noteLength
        for noteBool in tokens[2:14]:
            if noteBool == "YES":
                mid.tracks[0].append(Message('note_off', note=noteValue, velocity=vel, time=noteLen*1000))
                noteLen = 0 # once we set noteoff once, we only do it again with time = 0
            noteValue += 1
        mid.tracks[0].append(Message('note_off', note=bassNoteValue, velocity=bassVel, time=0))

    @staticmethod
    def arrayToMidi(choraleArray, midiName):
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        noteStart = 0
        noteLength = 4
        for chord in choraleArray:
            MusicHandler.AddChordToMidi(mid, chord, noteLength)
            noteStart += noteLength
        mid.save(midiName + '.mid')

    @staticmethod
    def AddChordToMidi(mid, chord, noteLength):
        vel = 40
        ### notes on -----------------------
        noteValue = 48
        for note in chord:
            if note == 1:
                mid.tracks[0].append(Message('note_on', note=noteValue, velocity=vel, time=0))
            noteValue+=1

        ### notes off ----------------------
        noteValue = 48
        noteLen = noteLength
        for note in chord:
            if note == 1:
                mid.tracks[0].append(Message('note_off', note=noteValue, velocity=vel, time=noteLen*1000))
                noteLen = 0 # once we set noteoff once, we only do it again with time = 0
            noteValue += 1



'''
def main():
    MusicHandler.getChoraleMidi(3,'jsbach_chorals_harmony.data')

main()
'''
