import tensorflow as tf
import numpy as np
import pretty_midi
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
import collections
from typing import Optional

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# sampling rate for audio playback
_SAMPLING_RATE = 16000

get_note_names = (pretty_midi.note_name_to_number)
get_note_numbers = (pretty_midi.note_number_to_name)

def transform_raw_data(l) -> pd.DataFrame:
  notes = collections.defaultdict(list)
  time_since_prev = 0
  prev_start = 0
  for note in l:
    if note['note']=="RE": # if is a rest
      time_since_prev += note['duration']
      continue
    pitch = get_note_names(note['note'])
    duration = note['duration']
    step = time_since_prev

    start = prev_start + step
    end = start + duration

    notes['pitch'].append(pitch)
    notes['duration'].append(duration)
    notes['step'].append(step)
    notes['start'].append(start)
    notes['end'].append(end)
    time_since_prev = duration
    prev_start = start

  return pd.DataFrame({name: value for name, value in notes.items()})

def notes_to_midi(
    notes: pd.DataFrame, #2d tabular data
    out_file: str,
    velocity:int=100, # note volume
) -> pretty_midi.PrettyMIDI():
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
  )
  prev_start = 0
  for i, note in notes.iterrows():
    start = prev_start+note['step']
    end = start+note['duration']
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
  waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
  # take a sample of the generated waveform to mitigate kernel resets
  waveform_short = waveform[:seconds*_SAMPLING_RATE]
  return display.Audio(waveform_short, rate=_SAMPLING_RATE)

def convert_to_dict(l):
  d = []
  for a in l:
    d.append({'note':a[0], 'duration':a[1]})
  return d

def plot_piano_roll(notes: pd.DataFrame, count:Optional[int]=None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20,4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color='b', marker='.'
  )
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)