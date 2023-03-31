import tensorflow as tf
import numpy as np

# Define the input and output sizes
input_size = 5
output_size = 10

# Generate some dummy data with variable-length sequences
sequences = [[np.random.randn(np.random.randint(10, 20), input_size) for _ in range(5)] for _ in range(100)]

# Pad the sequences to a fixed length
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, padding='post', dtype='float32')

# Create a binary mask to ignore padded values
mask = tf.cast(padded_sequences != 0, dtype='float32')

# Define the RNN model with GRU layer and masked input
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0., input_shape=(None, input_size)),
    tf.keras.layers.GRU(units=output_size),
    tf.keras.layers.Dense(units=output_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate some dummy labels for training
labels = np.random.randn(100, output_size)

# Train the model with masked input and sequence length
model.fit(padded_sequences, labels, epochs=10, sample_weight=mask)
