import numpy as np
import tensorflow as tf
from songs import alwayswithme
from tools import *
from dataset import *
from model import *

def predict_next_note(
    notes: np.ndarray,
    keras_model: tf.keras.Model,
    temperature: float = 1.0
) -> tuple[int, float, float]:
  assert temperature>0

  # add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = keras_model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # step and duration should be nonnegative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)

temperature = 2.0
num_predictions = 120

start_line_file_name = ""

d = alwayswithme
d = convert_to_dict(d)

starting_line = transform_raw_data(d)

sample_notes = np.stack([starting_line[key] for key in key_order], axis=1)

input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1])
)
generated_notes = []
prev_start = 0
prev_dur = 0
for _ in range(num_predictions):
  pitch, step, duration = predict_next_note(input_notes, model, temperature)
  #step += prev_dur
  start = prev_start + step
  end = start + duration
  input_note = (pitch, step, duration)
  generated_notes.append((*input_note, start, end))
  input_notes = np.delete(input_notes, 0, axis=0)
  input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
  prev_start = start
  prev_dur = duration

generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end')
)

plot_piano_roll(generated_notes)
out_file = 'output.mid'
out_pm = notes_to_midi(
    generated_notes, out_file=out_file)
display_audio(out_pm)
