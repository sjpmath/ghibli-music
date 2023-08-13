import tensorflow as tf
from model import model, loss, optimizer
from dataset import train_ds

model.compile(loss=loss,
              loss_weights={ # weight the losses because pitch loss tends to be high
                'pitch': 1.0,
                'step': 0.1,
                'duration': 0.5
              },
              optimizer=optimizer)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
]

epochs = 200
history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
)