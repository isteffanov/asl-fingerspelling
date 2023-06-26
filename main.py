import json
from parquet import load_parquet_as_np
from keras.utils import pad_sequences
from utils import *
import numpy as np

print("----------PREPARING DATA FROM FILE-----------")

input_parquet = '/home/asl-fingerspelling/preprocessed_train-2.parquet'
supplementary_file = 'train.csv'

other_parquet = '1647220008.parquet'

print(f"file: {input_parquet}")
print(f"supplementary file: {supplementary_file}")

sequences, phrases = load_parquet_as_np(input_parquet, supplementary_file)
sequences = sequences[0:40000]
phrases = phrases[0:40000]

ohe_phrases = np.array([phrase_to_ohe(phrase) for phrase in phrases])
ohe_p_start_and_end = np.array([add_start_and_end(phrase) for phrase in ohe_phrases])

print("Padding sequences...")

sequences = np.array([remove_nans(arr) for arr in sequences])
padded_sequences = pad_sequences(sequences,padding='post', dtype='float32')
ohe_p_start_and_end = pad_sequences(ohe_p_start_and_end,padding='post', dtype='float32')
ohe_phrases = pad_sequences(ohe_phrases,padding='post', dtype='float32')

print("Sequences padded")

print("----------SUCCESSFULY READ DATA FROM FILE-----------")

#### MODEL
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, AveragePooling1D, Conv1D
from keras.models import Model

print("----------COMPILING MODEL-----------")

encoder_inputs = Input(shape=(None, 126))

# Didn't improve much
#avg_pooling = AveragePooling1D(pool_size=126, padding='same', input_shape=(None, 126))(encoder_inputs)

# Didn't improve much
#conv = Conv1D(32, 3, activation='relu', input_shape=(None, 126))(encoder_inputs)

encoder = LSTM(128, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder2 = LSTM(128, return_sequences=True, return_state=True)
encoder_outputs2, state_h2, state_c2 = encoder2(encoder_outputs)

encoder3 = LSTM(128, return_sequences=True, return_state=True)
encoder_outputs3, state_h3, state_c3 = encoder3(encoder_outputs2)

decoder_inputs = Input(shape=(None, 61))
decoder = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h3, state_c3])

decoder_dense = Dense(61, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

from keras.utils.vis_utils import plot_model
plot_model(model)

import sys;sys.exit()

model.compile(optimizer="adam", loss="mse", metrics=["accuracy", "cosine_similarity"])

print("----------SUCCESSFULLY COMPILED MODEL-----------")

#encoder_input_data, decoder_input_data, decoder_target_data = prepare_data(sequences,phrases, char_to_num_map)

batch_size = 10
epochs = 50

def data_generator(padded_sequences, ohe_p_start_and_end, batch_size):
    num_samples = len(padded_sequences)
    while True:
        # Shuffle the data for each epoch
        indices = np.random.permutation(num_samples)
        padded_sequences = padded_sequences[indices]
        ohe_p_start_and_end = ohe_p_start_and_end[indices]

        # Generate batches
        for i in range(0, num_samples, batch_size):
            batch_padded_sequences = padded_sequences[i:i+batch_size]
            batch_ohe_p_start_and_end = ohe_p_start_and_end[i:i+batch_size]

            yield [batch_padded_sequences, batch_ohe_p_start_and_end], batch_ohe_p_start_and_end

#def data_generator(padded_sequences, ohe_p_start_and_end, batch_size, validation_split=0.2):
#    num_samples = len(padded_sequences)
#    num_train_samples = int(num_samples * (1 - validation_split))
#    indices = np.random.permutation(num_samples)
#    padded_sequences = padded_sequences[indices]
#    ohe_p_start_and_end = ohe_p_start_and_end[indices]
#
#    train_padded_sequences = padded_sequences[:num_train_samples]
#    train_ohe_p_start_and_end = ohe_p_start_and_end[:num_train_samples]
#    val_padded_sequences = padded_sequences[num_train_samples:]
#    val_ohe_p_start_and_end = ohe_p_start_and_end[num_train_samples:]
#
#    while True:
#        # Shuffle the training data for each epoch
#        train_indices = np.random.permutation(num_train_samples)
#        train_padded_sequences = train_padded_sequences[train_indices]
#        train_ohe_p_start_and_end = train_ohe_p_start_and_end[train_indices]
#
#        # Generate batches for training set
#        for i in range(0, num_train_samples, batch_size):
#            batch_padded_sequences = train_padded_sequences[i:i+batch_size]
#            batch_ohe_p_start_and_end = train_ohe_p_start_and_end[i:i+batch_size]
#
#            yield [batch_padded_sequences, batch_ohe_p_start_and_end], batch_ohe_p_start_and_end
#
#        # Generate batches for validation set
#        for i in range(0, len(val_padded_sequences), batch_size):
#            batch_padded_sequences = val_padded_sequences[i:i+batch_size]
#            batch_ohe_p_start_and_end = val_ohe_p_start_and_end[i:i+batch_size]
#
#            yield [batch_padded_sequences, batch_ohe_p_start_and_end], batch_ohe_p_start_and_end


#model.fit([padded_sequences, ohe_p_start_and_end], ohe_p_start_and_end, epochs=epochs, validation_split=0.2, batch_size=batch_size)

##### MANUAL VALIDATION
epochs = 10
batch_size = 64

generator = data_generator(padded_sequences[0:30000], ohe_p_start_and_end[0:30000], batch_size)

model.fit(generator, epochs=epochs, steps_per_epoch=len(padded_sequences[0:30000])//batch_size)

import pdb;pdb.set_trace()

val_generator = data_generator(padded_sequences[30000:40000], ohe_p_start_and_end[30000:40000], batch_size)

result = model.evaluate(val_generator, steps=500, return_dict=True)
print(result)
#####

