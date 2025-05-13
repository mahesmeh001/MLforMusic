#!/usr/bin/env python
# coding: utf-8

# ## Homework 3: Symbolic Music Generation Using Markov Chains

# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# In[1]:


# run this command to install MiDiTok
# ! pip install miditok


# In[2]:


# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
# from miditoolkit import MidiFile
from mido import MidiFile

from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile


# In[3]:


# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)


# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509). 
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# In[4]:


midi_files = glob('PDMX_subset/*.mid')
len(midi_files)


# ### Train a tokenizer with the REMI method in MidiTok

# In[5]:


config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)


# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`; a `Bar_None` token indicates the beginning of a new bar.

# In[6]:


# e.g.:
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
tokens[:10]


# 1. Write a function to extract note pitch events from a midi file; and another extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events (e.g. [60, 62, 61, ...])
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# In[7]:


get_ipython().system('pip install mido')


# In[8]:


def note_extraction(midi_file):
    # Q1a: Your code goes here
    note_events = []
    midi = MidiFile(midi_file)
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_events.append(msg.note)
                
    return note_events


# In[9]:


def note_frequency(midi_files):
    # Q1b: Your code goes here
    
    noteMap = defaultdict(int)
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for note in notes:
            noteMap[note] += 1
        
    return noteMap


# 2. Write a function to normalize the above dictionary to produce probability scores (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities, e.g. {60: 0.13, 61: 0.06, …}

# In[10]:


def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    unigramProbabilities = {}
    
    # Q2: Your code goes here
    # ...
    total_notes = sum(note_counts.values())
    for note, count in note_counts.items():
        unigramProbabilities[note] = count / total_notes
    
    return unigramProbabilities


# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) values for the dataset; write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key: previous_note, value: a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...} (i.e., this is a list of every other note that occured after note 60, every note that occured after note 62, etc.)
# 
#   - `bigramTransitionProbabilities`: key:previous_note, value: a list of probabilities for next_note in the same order of `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...} (i.e., you are converting the values above to probabilities)
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# In[11]:


def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    # Q3a: Your code goes here
    # ...
        
    bigramCounts = defaultdict(dict)
    
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for i in range(len(notes)-1):
            prev_note = notes[i]
            next_note = notes[i + 1]
            bigramCounts[prev_note][next_note] += 1
    
    for prev_note, next_note_dict in bigramCounts.items():
        next_notes = []
        counts = []
        total = 0 
    
        for next_note, count in next_note_dict.items():
            next_notes.append(next_note)
            counts.append(count)
            total += count
    
        probs = [count / total for count in counts]
        bigramTransitions[prev_note] = next_notes
        bigramTransitionProbabilities[prev_note] = probs

    return bigramTransitions, bigramTransitionProbabilities


# In[12]:


def sample_next_note(note):
    # Q3b: Your code goes here
    bigramTransitions, bigramTransitionProbabilities= note_bigram_probability(midi_files)
    
    if note not in bigramTransitions:
        return None  # fallback behavior
    next_notes = bigramTransitions[note]
    probabilities = bigramTransitionProbabilities[note]
    return random.choices(next_notes, weights=probabilities, k=1)[0]


# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# In[13]:


def note_bigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    
    # Q4: Your code goes here
    # Can use regular numpy.log (i.e., natural logarithm)
    
    notes = note_extraction(midi_file)
    if len(notes)<2:
        return float('inf')
    
    log_prob_sum = 0.0
    N = len(notes)    
    
    first_note = notes[0]
    p_first = unigramProbabilities[first_note]
    log_prob_sum += np.log(p_first)
    
    for i in range(1,N):
        prev_note = notes[i-1]
        curr_note = notes[i]
        
        next_notes = bigramTransitions[prev_note]
        probs = bigramTransitionProbabilities.get(prev_note, [])

        if curr_note in next_notes:
            idx = next_notes.index(curr_note)
            p = probs[idx]
        else:
            p = 1e-6

        log_prob_sum += np.log(p)

    perplexity = np.exp(-log_prob_sum / N)
    return perplexity
    
    
    


# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); write a function to compute the perplexity of this new model on a midi file. 
# 
#     The perplexity of this model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key: (next_previous_note, previous_note), value: a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# In[14]:


def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)
    
    # Q5a: Your code goes here
    # ...
    
    trigramTransitionCounts = defaultdict(dict)
    
    
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for i in range(2, len(notes)):
            prev_prev = notes[i - 2]
            prev = notes[i - 1]
            curr = notes[i]

            key = (prev_prev, prev)
            if curr in trigramTransitionCounts[key]:
                trigramTransitionCounts[key][curr] += 1
            else:
                trigramTransitionCounts[key][curr] = 1

    # Now convert counts to transitions and probabilities
    for key in trigramTransitionCounts:
        next_notes = list(trigramTransitionCounts[key].keys())
        counts = list(trigramTransitionCounts[key].values())
        total = sum(counts)
        probs = [count / total for count in counts]

        trigramTransitions[key] = next_notes
        trigramTransitionProbabilities[key] = probs

    return trigramTransitions, trigramTransitionProbabilities


# In[15]:


def note_trigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Q5b: Your code goes here
    
    notes = note_extraction(midi_file)
    if len(notes) < 3:
        return float('inf')

    log_prob_sum = 0.0
    N = len(notes)
    epsilon = 1e-10  # small fallback value

    # First note uses unigram
    p1 = unigramProbabilities.get[notes[0]]
    log_prob_sum += np.log(p1)

    # Second note uses bigram
    prev = notes[0]
    curr = notes[1]
    next_notes = bigramTransitions[prev]
    probs = bigramTransitionProbabilities[prev]

    if curr in next_notes:
        idx = next_notes.index(curr)
        p2 = probs[idx]
    else:
        p2 = epsilon
    log_prob_sum += np.log(p2)

    # From third note onwards use trigram
    for i in range(2, N):
        prev_prev = notes[i - 2]
        prev = notes[i - 1]
        curr = notes[i]

        key = (prev_prev, prev)
        next_notes = trigramTransitions.get(key, [])
        probs = trigramTransitionProbabilities.get(key, [])

        if curr in next_notes:
            idx = next_notes.index(curr)
            p = probs[idx]
        else:
            p = epsilon

        log_prob_sum += np.log(p)

    perplexity = np.exp(-log_prob_sum / N)
    return perplexity


# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# In[16]:


duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}


# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# In[17]:


def beat_extraction(midi_file):
    # Q6: Your code goes here
    
    midi = MidiFile(midi_file)    
    tokenizer = REMI()
    tokens = tokenizer(midi)
    
    output = []
    current_pos = None
    for token in tokens:
        if token.type == "Position":
            current_pos = int(token.value)
        elif token.type == "Duration":
            beat_length = duration2length.get(token.value)
            if current_pos is not None and beat_length is not None:
                output.append((current_pos % 32, beat_length))
                current_pos = None  # Reset for next note
    return output


# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key: previous_beat_length, value: a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# In[18]:


def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)
    
    # Q7: Your code goes here
    # ...
    for midi_file in midi_files:
        # Extract beat positions and lengths for each midi file
        beat_info = beat_extraction(midi_file)  # Assuming this gives [(position, beat_length), ...]
        
        # Iterate through the beat lengths and track transitions
        for i in range(1, len(beat_info)):
            previous_beat_length = beat_info[i - 1][1]  # Get previous beat length
            current_beat_length = beat_info[i][1]  # Get current beat length
            
            # Track the transitions
            bigramBeatTransitions[previous_beat_length].append(current_beat_length)
    
    # Now calculate probabilities
    for previous_beat_length, next_beat_lengths in bigramBeatTransitions.items():
        total_transitions = len(next_beat_lengths)
        transition_counts = defaultdict(int)
        
        for next_beat_length in next_beat_lengths:
            transition_counts[next_beat_length] += 1
        
        # Calculate probabilities based on counts
        probabilities = [count / total_transitions for count in transition_counts.values()]
        
        bigramBeatTransitionProbabilities[previous_beat_length] = probabilities
        # Ensure the list of possible next beat lengths is also in the same order as probabilities
        bigramBeatTransitions[previous_beat_length] = list(transition_counts.keys())
    
    return bigramBeatTransitions, bigramBeatTransitionProbabilities


# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# In[19]:


def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)
    
    # Q8a: Your code goes here
    # ...
    
    for midi_file in midi_files:
        # Extract beat position and length pairs from the midi file
        beat_info = beat_extraction(midi_file)  # [(position, beat_length), ...]
        
        for i in range(1, len(beat_info)):
            previous_beat_position = beat_info[i - 1][0]  # Beat position
            current_beat_length = beat_info[i][1]  # Beat length

            # Track transitions for (beat_position -> beat_length)
            bigramBeatPosTransitions[previous_beat_position].append(current_beat_length)
    
    # Calculate transition probabilities
    for previous_beat_position, next_beat_lengths in bigramBeatPosTransitions.items():
        total_transitions = len(next_beat_lengths)
        transition_counts = defaultdict(int)

        for next_beat_length in next_beat_lengths:
            transition_counts[next_beat_length] += 1
        
        probabilities = [count / total_transitions for count in transition_counts.values()]
        
        bigramBeatPosTransitionProbabilities[previous_beat_position] = probabilities
        bigramBeatPosTransitions[previous_beat_position] = list(transition_counts.keys())
    
    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities


# In[20]:


import math


def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    # Q8b: Your code goes here
    # Hint: one more probability function needs to be computed
    # Helper function to calculate perplexity given a model
    def calculate_perplexity(transitions, transition_probabilities, beat_info):
        log_prob_sum = 0
        N = len(beat_info)
        
        for i in range(1, N):
            prev_value = beat_info[i - 1][1]  # Get previous beat length
            current_value = beat_info[i][1]  # Get current beat length
            
            if prev_value in transitions:
                possible_next_values = transitions[prev_value]
                probabilities = transition_probabilities[prev_value]
                
                if current_value in possible_next_values:
                    idx = possible_next_values.index(current_value)
                    prob = probabilities[idx]
                else:
                    prob = 1e-6  # Small probability if not found in transitions
            else:
                prob = 1e-6  # Small probability if previous value has no transition
            
            log_prob_sum += math.log(prob)
        
        return math.exp(-log_prob_sum / N)
    
    beat_info = beat_extraction(midi_file)  # [(position, beat_length), ...]


    # perplexity for Q7
    perplexity_Q7 = calculate_perplexity(bigramBeatTransitions, bigramBeatTransitionProbabilities, beat_info)
    
    # perplexity for Q8
    perplexity_Q8 = calculate_perplexity(bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities, beat_info)
    
    return perplexity_Q7, perplexity_Q8


# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity. 
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key: (previous_beat_length, beat_position), value: a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramBeatTransitions`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# In[21]:


def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)

    for midi_file in midi_files:
        # Extract beat position and length pairs from the midi file
        beat_info = beat_extraction(midi_file)  # [(position, beat_length), ...]
        
        for i in range(2, len(beat_info)):  # Start from index 2 to have previous two values
            previous_beat_length = beat_info[i - 2][1]  # Previous beat length
            current_beat_position = beat_info[i - 1][0]  # Current beat position
            current_beat_length = beat_info[i][1]  # Current beat length

            # Track transitions for (previous_beat_length, beat_position -> beat_length)
            trigramBeatTransitions[(previous_beat_length, current_beat_position)].append(current_beat_length)

    # Calculate transition probabilities
    for (previous_beat_length, current_beat_position), next_beat_lengths in trigramBeatTransitions.items():
        total_transitions = len(next_beat_lengths)
        transition_counts = defaultdict(int)

        for next_beat_length in next_beat_lengths:
            transition_counts[next_beat_length] += 1
        
        probabilities = [count / total_transitions for count in transition_counts.values()]
        
        trigramBeatTransitionProbabilities[(previous_beat_length, current_beat_position)] = probabilities
        trigramBeatTransitions[(previous_beat_length, current_beat_position)] = list(transition_counts.keys())
    
    return trigramBeatTransitions, trigramBeatTransitionProbabilities


# In[22]:


def beat_trigram_perplexity(midi_file):
    # Q7: Get the beat length model (previous model from Q7)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    
    # Q9a: Get the trigram-based model
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    
    # Helper function to calculate perplexity given a model
    def calculate_perplexity(transitions, transition_probabilities, beat_info):
        log_prob_sum = 0
        N = len(beat_info)
        
        for i in range(2, N):
            prev_beat_length = beat_info[i - 2][1]  # Previous beat length
            current_beat_position = beat_info[i - 1][0]  # Current beat position
            current_beat_length = beat_info[i][1]  # Current beat length
            
            if (prev_beat_length, current_beat_position) in transitions:
                possible_next_values = transitions[(prev_beat_length, current_beat_position)]
                probabilities = transition_probabilities[(prev_beat_length, current_beat_position)]
                
                if current_beat_length in possible_next_values:
                    idx = possible_next_values.index(current_beat_length)
                    prob = probabilities[idx]
                else:
                    prob = 1e-6  # Small probability if not found in transitions
            else:
                prob = 1e-6  # Small probability if previous value has no transition
            
            log_prob_sum += math.log(prob)
        
        return math.exp(-log_prob_sum / N)
    
    # Extract beat information from the midi file
    beat_info = beat_extraction(midi_file)  # [(position, beat_length), ...]
    
    # Perplexity for Q9 (Trigram Model)
    perplexity_Q9 = calculate_perplexity(trigramBeatTransitions, trigramBeatTransitionProbabilities, beat_info)
    
    return perplexity_Q9


# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# In[23]:


def music_generate(length):
    # sample notes
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Initialize the sampled notes list with a starting note (e.g., a random note from the unigram model)
    sampled_notes = [random.choice(list(unigramProbabilities.keys()))]  # Start with a random note
    
    # Sample notes based on the trigram model
    for i in range(2, length):  # Start from index 2 because we need at least 2 notes to form a trigram
        prev_prev = sampled_notes[i - 2]
        prev = sampled_notes[i - 1]
        
        key = (prev_prev, prev)
        if key in trigramTransitions:
            next_notes = trigramTransitions[key]
            probs = trigramTransitionProbabilities[key]
            sampled_notes.append(random.choices(next_notes, probs)[0])
        else:
            sampled_notes.append(random.choice(list(unigramProbabilities.keys())))  # Fall back to unigram
    
    # Step 2: Sample beat lengths using the trigram model (Q8)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    
    # Initialize the sampled beats list
    sampled_beats = []
    current_beat_position = 0
    previous_beat_length = random.choice(list(trigramBeatTransitions.keys()))[0]  # Start with a random previous beat length
    
    for i in range(length):
        key = (previous_beat_length, current_beat_position)
        if key in trigramBeatTransitions:
            next_beat_lengths = trigramBeatTransitions[key]
            probs = trigramBeatTransitionProbabilities[key]
            beat_length = random.choices(next_beat_lengths, probs)[0]
        else:
            beat_length = random.choice(list(trigramBeatTransitions.values()))[0]  # Fall back to random beat length

        sampled_beats.append(beat_length)
        
        # Update the beat position, and reset to 0 if the end of a bar is reached (assuming 4 beats per bar)
        current_beat_position += 1
        if current_beat_position == 4:
            current_beat_position = 0
        
        # Update previous beat length for next iteration
        previous_beat_length = beat_length
    
    # Step 3: Save generated music as a MIDI file (q10.mid)
    midi = MIDIFile(1)  # Create a new MIDI file with one track
    track = 0
    midi.addTrackName(track, 0, "Generated Music")
    midi.addTempo(track, 0, 120)  # Set the tempo (beats per minute)
    
    time = 0  # Start time in beats
    for note, beat_length in zip(sampled_notes, sampled_beats):
        midi.addNote(track, 0, note, time, beat_length / 8, 100)  # Divide beat length by 8 for MIDIUtil
        time += beat_length / 8  # Increment the time by the length of the note in beats
    
    with open("q10.mid", "wb") as f:
        midi.writeFile(f)


# In[24]:


get_ipython().system('jupyter nbconvert homework3.ipynb --to python')

