# Can be used for storing MIDI object files

class MIDIFILE(object):
    def __init__(self, name, start, end, pitch, velocity):
        # Store MIDI file attributes
        self.name = name
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)