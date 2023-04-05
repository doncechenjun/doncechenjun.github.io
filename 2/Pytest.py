import numpy as np
import tensorflow as tf

# Set up input and output shapes
input_shape = (None, 5)
output_shape = (None, 10)

# Generate random input and output data
input_data = np.array([np.random.rand(np.random.randint(1, 100), 5).astype(np.float32) for _ in range(1000)])
#output_data = np.array([np.random.rand(input_.shape[0], 10).astype(np.float32) for input_ in input_data])

# Pad the input sequences with zeros
input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, padding='post', dtype='float32')
output_data = np.array([np.random.rand(input_.shape[0], 10).astype(np.float32) for input_ in input_data])


# Create a sequential model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Masking(mask_value=0., input_shape=input_shape))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True))
model.add(tf.keras.layers.Dense(output_shape[1]))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
history=model.fit(input_data, 
          output_data, 
          epochs=10, 
          batch_size=32,
          validation_split=0.2
          )
