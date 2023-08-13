from songs import *
from tools import *

seq_length = 8
vocab_size = 128 # all possible notes on piano
batch_size = 64


def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size=128,
) -> tf.data.Dataset:
  seq_length = seq_length+1

  # take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

  # flat_map flattens the dataset of datasets into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size, 1.0, 1.0]
    return x

  # split labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}
    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

songlist = [alwayswithme, onesummersday, merrygoround, reprise, flowingclouds, mononoke, laputa, ashitakasan, kiki, promiseofworld, porco]
wholelist = [note for song in songlist for note in song]
wholelist = convert_to_dict(wholelist)

rawdata = transform_raw_data(wholelist)

all_notes = rawdata

key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
seq_ds.element_spec

buffer_size = len(all_notes)-seq_length # no. items in the dataset

train_ds = (seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))