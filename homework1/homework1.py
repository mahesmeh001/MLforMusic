#!/usr/bin/env python
# coding: utf-8

# # Homework 1: Sine wave generation and binary classification

# ## Part A - Sine Wave Generation

# ### Setup
# To complete this part, install the required Python libraries:

# In[18]:


import numpy as np
from scipy.io import wavfile

import numpy as np
import glob
from mido import MidiFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[19]:


# (installation process may be different on your system)
# You don't need to use these libraries, so long as you implement the specified functions
# !pip install numpy
# !pip install scipy
# !pip install IPython
# !pip install glob
# !pip install scikit-learn
# !pip install mido


# 1. Write a function that converts a musical note name to its corresponding frequency in Hertz (Hz)
# 
# `note_name_to_frequency()`
# - **Input**: A string `note_name` combining a note (e.g., `'C'`, `'C#'`, `'D'`, `'D#'`, `'E'`, `'F'`, `'F#'`, `'G'`, `'G#'`, `'A'`, `'A#'`, `'B'`) and an octave number (`'0'` to `'10'`)
# - **Output**: A float representing the frequency in Hz
# - **Details**:
#   - Use A4 = 440 Hz as the reference frequency
#   - Frequencies double with each octave increase (e.g., A5 = 880 Hz) and halve with each decrease (e.g., A3 = 220 Hz)
# 
# - **Examples**:
#   - `'A4'` → `440.0`
#   - `'A3'` → `220.0`
#   - `'G#4'` → `415.3047`

# In[20]:


SAMPLE_RATE = 44100
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 
             'F#', 'G', 'G#', 'A', 'A#', 'B']
def note_name_to_frequency(note_name):
    import math

    # Reference: A4 = 440 Hz
    A4_FREQ = 440.0
    A4_INDEX = 9 + 12 * 4  # A is the 10th note (index 9) in the chromatic scale, octave 4
    
    # Extract the note and octave from the input
    for i in range(1, len(note_name)):
        if note_name[i].isdigit():
            note = note_name[:i]
            octave = int(note_name[i:])
            break

    if note not in NOTES:
        raise ValueError(f"Invalid note name: {note}")

    note_index = NOTES.index(note)
    semitone_diff = (octave * 12 + note_index) - A4_INDEX

    # Compute frequency using the formula:
    frequency = A4_FREQ * (2 ** (semitone_diff / 12))
    return round(frequency, 4)


# 2. Write a function that linearly decreases the amplitude of a given waveform
# 
# `decrease_amplitude()`
# - **Inputs**:
#   - `audio`: A NumPy array representing the audio waveform at a sample rate of 44100 Hz
# - **Output**: A NumPy array representing the audio waveform at a sample rate of 44100 Hz
# - **Details**:
#   - The function must linearly decrease the amplitude of the input audio. The amplitude should start at 1 (full volume) and decrease gradually to 0 (silence) by the end of the sample

# In[21]:


def decrease_amplitude(audio):
    # Q2: Your code goes here
    fade = np.linspace(1, 0, len(audio))
    return audio * fade


# 3. Write a function that adds a delay effect to a given audio where the output is a combination of the original audio and a delayed audio
# 
# `add_delay_effects()`  
# - **Inputs**:  
#   - `audio`: A NumPy array representing the audio waveform, sampled at 44,100 Hz
# - **Output**:  
#   - A NumPy array representing the modified audio waveform, sampled at 44,100 Hz
# - **Details**:
#   - The amplitude of the delayed audio should be 30% of the original audio's amplitude
#   - The amplitude of the original audio should be adjusted to 70% of the original audio's amplitude
#   - The output should combine the original audio (with the adjusted amplitude) with a delayed version of itself
#   - The delayed audio should be offset by 0.5 seconds behind the original audio
# 
# - **Examples**:
#   - The provided files (input.wav and output.wav) provide examples of input and output audio

# In[22]:


# Can use these for visualization if you like, though the autograder won't use ipython
#
# from IPython.display import Audio, display
#
# print("Example Input Audio:")
# display(Audio(filename = "input.wav", rate=44100))
# 
# print("Example Output Audio:")
# display(Audio(filename = "output.wav", rate=44100))


# In[23]:


def add_delay_effects(audio, sample_rate = 44100):
    #Q3: Your code goes here
    delay_time = 0.5
    delay = int(delay_time * sample_rate)
    
    delayed_audio = np.zeros(len(audio) + delay)
    delayed_audio[delay:] += 0.3 * audio # delayed at 30%
    delayed_audio[:len(audio)] += 0.7 * audio # original at 70%
    return delayed_audio


# 4. Write a function that concatenates a list of audio arrays sequentially and a function that mixes audio arrays by scaling and summing them, simulating simultaneous playback
# 
# `concatenate_audio()`
# - **Input**:
#   - `list_of_your_audio`: A list of NumPy arrays (e.g., `[audio1, audio2]`), each representing audio at 44100 Hz
# - **Output**: A NumPy array of the concatenated audio
# - **Example**:
#   - If `audio1` is 2 seconds (88200 samples) and `audio2` is 1 second (44100 samples), the output is 3 seconds (132300 samples)
# 
# `mix_audio()`
# - **Inputs**:
#   - `list_of_your_audio`: A list of NumPy arrays (e.g., `[audio1, audio2]`), all with the same length at 44100 Hz.
#   - `amplitudes`: A list of floats (e.g., `[0.2, 0.8]`) matching the length of `list_of_your_audio`
# - **Output**: A NumPy array representing the mixed audio
# - **Example**:
#   - If `audio1` and `audio2` are 2 seconds long, and `amplitudes = [0.2, 0.8]`, the output is `0.2 * audio1 + 0.8 * audio2`

# In[24]:


def concatenate_audio(list_of_your_audio):
    #Q4: Your code goes here
    # just iterate through and sum each element?
    return np.concatenate(list_of_your_audio)


# In[25]:


def mix_audio(list_of_your_audio, amplitudes):
    #Q4: Your code goes here
    # sum but according to the amplitude...
    length = len(list_of_your_audio[0])
    mixed_audio = np.zeros(length)
    for audio, amp in zip(list_of_your_audio, amplitudes):
        mixed_audio += amp*audio
    return mixed_audio


# 5. Modify your solution to Q2 so that your pipeline can generate sawtooth waves by adding harmonics based on the following equation:
# 
#     $\text{sawtooth}(f, t) = \frac{2}{\pi} \sum_{k=1}^{19} \frac{(-1)^{k+1}}{k} \sin(2\pi k f t)$ 
# 
# - **Inputs**:
#   - `frequency`: Fundamental frequency of sawtooth wave
#   - `duration`: A float representing the duration in seconds (e.g., 2.0)
# - **Output**: A NumPy array representing the audio waveform at a sample rate of 44100 Hz

# In[26]:


def create_sawtooth_wave(frequency, duration, sample_rate=44100):
    #Q5: Your code goes here 
    # basically use sawtooth instead of linear 
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    wave = np.zeros_like(t)
    for k in range(1, 20):  # k from 1 to 19
        wave += ((-1) ** (k + 1)) * (1 / k) * np.sin(2 * np.pi * k * frequency * t)
    
    wave *= 2 / np.pi

    # Apply linearly decreasing amplitude envelope
    fade = np.linspace(1, 0, len(wave))
    return wave * fade
    return wave


# ## Part B - Binary Classification
# Train a binary classification model using `scikit-learn` to distinguish between piano and drum MIDI files.

# #### Unzip MIDI Files
# Extract the provided MIDI datasets:
# 
# ```bash
# unzip piano.zip
# unzip drums.zip
# ```
# 
# - `./piano`: Contains piano MIDI files (e.g., `0000.mid` to `2154.mid`)
# - `./drums`: Contains drum MIDI files (e.g., `0000.mid` to `2154.mid`)
# - Source: [Tegridy MIDI Dataset] (https://github.com/asigalov61/Tegridy-MIDI-Dataset)
# 
# These folders should be extracted into the same directory as your solution file

# 6. Write functions to compute simple statistics about the files
# 
# ####  `get_stats()`
# 
# - **Inputs**:
#   - `piano_file_paths`: List of piano MIDI file paths`
#   - `drum_file_paths`: List of drum MIDI file paths`
# - **Output**: A dictionary:
#   - `"piano_midi_num"`: Integer, number of piano files
#   - `"drum_midi_num"`: Integer, number of drum files
#   - `"average_piano_beat_num"`: Float, average number of beats in piano files
#   - `"average_drum_beat_num"`: Float, average number of beats in drum files
# - **Details**:
#   - For each file:
#     - Load with `MidiFile(file_path)`
#     - Get `ticks_per_beat` from `mid.ticks_per_beat`
#     - Compute total ticks as the maximum cumulative `msg.time` (delta time) across tracks
#     - Number of beats = (total ticks / ticks_per_beat)
#   - Compute averages, handling empty lists (return 0 if no files)

# In[27]:


def get_file_lists():
    piano_files = sorted(glob.glob("./piano/*.mid"))
    drum_files = sorted(glob.glob("./drums/*.mid"))
    return piano_files, drum_files

def get_num_beats(file_path):
    # Q6: Your code goes here
    mid = MidiFile(file_path)
    # Might need: mid.tracks, msg.time, mid.ticks_per_beat
    
    max_track_ticks = 0
    for track in mid.tracks:
        this_ticks = 0 
        for msg in track:
            this_ticks += msg.time
        max_track_ticks = max(max_track_ticks, this_ticks)
    
    return max_track_ticks/mid.ticks_per_beat if mid.ticks_per_beat else 0
    

def get_stats(piano_path_list, drum_path_list):
    piano_beat_nums = []
    drum_beat_nums = []
    for file_path in piano_path_list:
        piano_beat_nums.append(get_num_beats(file_path))
        
    for file_path in drum_path_list:
        drum_beat_nums.append(get_num_beats(file_path))
    
    return {"piano_midi_num":len(piano_path_list),
            "drum_midi_num":len(drum_path_list),
            "average_piano_beat_num":np.average(piano_beat_nums),
            "average_drum_beat_num":np.average(drum_beat_nums)}


# 7. Implement a few simple feature functions, to compute the lowest and highest MIDI note numbers in a file, and the set of unique notes in a file
# 
# `get_lowest_pitch()` and `get_highest_pitch()`
# functions to find the lowest and highest MIDI note numbers in a file
# 
# - **Input**: `file_path`, a string (e.g., `"./piano/0000.mid"`)
# - **Output**: An integer (0–127) or `None` if no notes exist
# - **Details**:
#   - Use `MidiFile(file_path)` and scan all tracks
#   - Check `msg.type == 'note_on'` and `msg.velocity > 0` for active notes
#   - Return the minimum (`get_lowest_pitch`) or maximum (`get_highest_pitch`) `msg.note`
# 
# `get_unique_pitch_num()`
# a function to count unique MIDI note numbers in a file
# 
# - **Input**: `file_path`, a string
# - **Output**: An integer, the number of unique pitches
# - **Details**:
#   - Collect `msg.note` from all `'note_on'` events with `msg.velocity > 0` into a set
#   - Return the set’s length
# - **Example**: For notes `["C4", "C4", "G4", "G4", "A4", "A4", "G4"]`, output is 3 (unique: C4, G4, A4)

# In[28]:


def get_lowest_pitch(file_path):
    #Q7-1: Your code goes here
    # go though tracks and messages to find note
    lowest_note = 0
    mid = MidiFile(file_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note < lowest_note:
                    lowest_note = msg.note
    
    return lowest_note

def get_highest_pitch(file_path):
    #Q7-2: Your code goes here
    highest_note = 0
    mid = MidiFile(file_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note > highest_note:
                    highest_note = msg.note
                    
    return highest_note

def get_unique_pitch_num(file_path):
    #Q7-3: Your code goes here
    mid = MidiFile(file_path)
    notes = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.add(msg.note)
    return len(notes)


# 8. Implement an additional feature extraction function to compute the average MIDI note number in a file
# 
# `get_average_pitch_value()`
# a function to return the average MIDI note number from a file
# 
# - **Input**: `file_path`, a string
# - **Output**: A float, the average value of MIDI notes in the file
# - **Details**:
#   - Collect `msg.note` from all `'note_on'` events with `msg.velocity > 0` into a set
# - **Example**: For notes `[51, 52, 53]`, output is `52`

# In[29]:


def get_average_pitch_value(file_path):
    #Q8: Your code goes here
    mid = MidiFile(file_path)
    
    notes = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
    
    if notes:
        return sum(notes) / len(notes)
    else:
        return None


# 9. Construct your dataset and split it into train and test sets using `scikit-learn` (most of this code is provided). Train your model to classify whether a given file is intended for piano or drums.
# 
# `featureQ9()`
# 
# Returns a feature vector concatenating the four features described above
# 
# - **Input**: `file_path`, a string.
# - **Output**: A vector of four features

# In[30]:


def featureQ9(file_path):
    # Already implemented: this one is a freebie if you got everything above correct!
    return [get_lowest_pitch(file_path),
            get_highest_pitch(file_path),
            get_unique_pitch_num(file_path),
            get_average_pitch_value(file_path)]


# 10. Creatively incorporate additional features into your classifier to make your classification more accurate.  Include comments describing your solution.

# In[31]:


def featureQ10(file_path):
    #Q10: Your code goes here
    # add a bias
    # todo: add... 0) a bias term 1) track tempo 2) track duration 3) stuff from get_stats
    return [1, get_lowest_pitch(file_path), get_highest_pitch(file_path), get_unique_pitch_num(file_path), get_average_pitch_value(file_path)]


# In[ ]:


# get_ipython().system('jupyter nbconvert homework1.ipynb --to python')


# In[33]:




