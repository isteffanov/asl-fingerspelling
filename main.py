import json
from parquet import load_parquet_as_np
from keras.utils import pad_sequences
import numpy as np

with open('character_to_prediction_index.json') as f:
    char_to_num_map = json.load(f)

num_to_char_map = {v: k for k, v in char_to_num_map.items()}

def char_to_num(char):
    if char == " ":
        return 0

    if not char_to_num_map.get(char):
        raise Exception(f'Character "{char}" not found in dict when translating char to num')
    else:
        return char_to_num_map[char]

def num_to_char(num):
    if not num_to_char_map.get(num):
        raise Exception(f'Num "{num}" not found in dict when translating num to char')
    else:
        return num_to_char_map[num]

def phrase_to_nums(phrase):
    arr = []
    for char in phrase:
        arr.append(char_to_num(char))

    return arr

def phrase_to_ohe(phrase):
    return np.array([char_to_ohe(c) for c in phrase])

def nums_to_phrase(arr):
    phrase = ''
    for num in arr:
        phrase += num_to_char(num)

    return phrase

def add_phrase_as_nums_to_sequences(sequences):
    for sequence in sequences:
        sequence.text_as_nums = phrase_to_nums(sequence.text)

def char_to_ohe(char):
    num = char_to_num(char)
    arr = np.zeros((61,), dtype=int)
    arr[num+1] = 1

    return arr

def start_token():
    arr = np.zeros((61,), dtype=int)
    arr[0] = 1
    return arr

def end_token():
    arr = np.zeros((61,), dtype=int)
    arr[60] = 1
    return arr

def add_start_and_end(arr):
    arr = np.append(arr, [end_token()], axis=0)
    arr = np.insert(arr, 0, [start_token()], axis=0)
    return arr

def remove_nans(arr):
    arr[np.isnan(arr)] = 0
    return arr


sequences, phrases = load_parquet_as_np('1647220008.parquet', 'train.csv')

import numpy as np

def prepare_data(input_data, target_texts, char_to_num_map):
    # Determine the maximum sequence length in the input data
    max_sequence_length = max(len(seq) for seq in input_data)

    # Determine the number of values in each frame of the input sequence
    num_values = input_data[0].shape[1]

    # Initialize the encoder input data array
    encoder_input_data = np.zeros((len(input_data), max_sequence_length, num_values), dtype=np.float32)

    # Iterate over each input sequence
    for i, input_sequence in enumerate(input_data):
        # Assign each frame to the corresponding position in the encoder_input_data array
        encoder_input_data[i, :input_sequence.shape[0], :] = input_sequence
        pdb.set_trace()

    # Determine the maximum target sequence length
    max_decoder_seq_length = max(len(target_text) for target_text in target_texts)

    # Initialize the decoder input and target data arrays
    num_decoder_tokens = len(char_to_num_map)
    decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.float32)
    decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.float32)

    # Iterate over each target text
    for i, target_text in enumerate(target_texts):
        # Iterate over each character in the target text
        for t, char in enumerate(target_text):
            # Set the corresponding position to 1.0 in the decoder input data
            decoder_input_data[i, t, char_to_num_map[char]] = 1.0

            if t > 0:
                # Set the corresponding position to 1.0 in the decoder target data
                decoder_target_data[i, t - 1, char_to_num_map[char]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data

ohe_phrases = np.array([phrase_to_ohe(phrase) for phrase in phrases])
ohe_p_start_and_end = np.array([add_start_and_end(phrase) for phrase in ohe_phrases])

#### MODEL
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

encoder_inputs = Input(shape=(None, 84))
encoder1 = LSTM(256, return_sequences=True, return_state=True)
encoder_outputs1, state_h1, state_c1 = encoder1(encoder_inputs)

#encoder2 = LSTM(500, return_sequences=True, return_state=True)
#encoder_outputs2, state_h2, state_c2 = encoder2(encoder_outputs1)
#
#encoder3 = LSTM(500, return_sequences=True, return_state=True)
#encoder_outputs3, state_h3, state_c3 = encoder3(encoder_outputs2)

decoder_inputs = Input(shape=(None, 61))

decoder = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h1, state_c1])

decoder_dense = Dense(61, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy", "cosine_similarity"])

encoder_input_data, decoder_input_data, decoder_target_data = prepare_data(sequences,phrases, char_to_num_map)

batch_size = 128 
epochs = 20

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.Accuracy(),
                       tf.keras.metrics.CosineSimilarity()])
sequences = np.array([remove_nans(arr) for arr in sequences])
padded_sequences = pad_sequences(sequences,padding='post', dtype='float32')
ohe_p_start_and_end = pad_sequences(ohe_p_start_and_end,padding='post', dtype='float32')
ohe_phrases = pad_sequences(ohe_phrases,padding='post', dtype='float32')
#model.fit([tf.ragged.constant(sequences), tf.ragged.constant(ohe_p_start_and_end)], tf.ragged.constant(ohe_phrases))
#model.fit([padded_sequences, ohe_p_start_and_end], ohe_phrases)
model.fit([padded_sequences, ohe_p_start_and_end], ohe_p_start_and_end, epochs=100, validation_split=0.2)

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
#### END OF MODEL
