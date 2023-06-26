import numpy as np
import pandas as pd
import os
from attention import AttentionLayer

class Sequence: 
    def __init__(self, df, text, seq_id, num_frames):
        # DataFrame
        self.df = df
        # Sequence Text
        self.text = text
        # seq id
        self.seq_id = seq_id
        
        self.frames = num_frames


def load_landmark_file(landmark_file, metadata_df):
    data = pd.read_parquet(landmark_file)
    group = data.groupby('sequence_id')
    
    sequences = []
    for idx, df in group:
        sequence_id = idx
        text = metadata_df[metadata_df['sequence_id'] == sequence_id].iloc[0].phrase
        num_frames = len(df)
        sequences.append(Sequence(
            df=df.filter(regex=r'[xyz]_(dom|sub)_[0-9]+'), \
            text=text, \
            seq_id=sequence_id, \
            num_frames=num_frames))
            
    return sequences


import pickle

if not os.path.exists('nns.pkl'):

    landmark_dir = '/home/asl-fingerspelling/preprocessed_train-2.parquet'
    csv_path = '/home/data/train.csv'

    metadata = pd.read_csv(csv_path)
    landmarks = pd.read_parquet(landmark_dir)

    sequences = pd.DataFrame(landmarks.groupby(['sequence_id']).apply(lambda x:x.to_numpy()))
    sequences = sequences.rename(columns={0: 'coordinates'})

    sequences = sequences.sort_values(by='sequence_id')

    phrases = pd.read_csv(csv_path, usecols=['sequence_id', 'phrase'])

    phrases = phrases.sort_values(by='sequence_id')
    matched_sequences = pd.merge(sequences,phrases, on = 'sequence_id')

    with open('nns.pkl', 'wb+') as file:
        pickle.dump(matched_sequences, file)

else:
    with open('nns.pkl', 'rb') as file:
        matched_sequences = pickle.load(file)


import json
from parquet import load_parquet_as_np
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
    arr = np.zeroes((61,), dtype=int)
    arr[0] = 1
    return arr

def end_token():
    arr = np.zeroes((61,), dtype=int)
    arr[60] = 1
    return arr


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


#### MODEL
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.models import Model


encoder_inputs = Input(shape=(None, 42))
encoder = LSTM(512, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#encoder_states = [state_h, state_c]

encoder2 = LSTM(512, return_sequences=True, return_state=True)
encoder_outputs2, state_h2, state_c2 = encoder2(encoder_outputs)

encoder3 = LSTM(512, return_sequences=True, return_state=True)
encoder_outputs3, state_h3, state_c3 = encoder3(encoder_outputs2)

decoder_inputs = Input(shape=(None, 59))
decoder = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h3, state_c3])

decoder_dense = Dense(59, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer="adam", loss="mse", metrics=["accuracy", "cosine_similarity"])

encoder_input_data, decoder_input_data, decoder_target_data = prepare_data([row["coordinates"] for _, row in matched_sequences.iterrows()],
    	[row["phrase"] for _, row in matched_sequences.iterrows()] , char_to_num_map)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split([encoder_input_data,decoder_input_data], decoder_input_data, test_size=0.33, random_state=42)

batch_size = 128
epochs = 50


model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
#### END OF MODEL

#### TEST

model.predict(X_test, batch_size, )
# RESULTS

# 165/165 [==============================] - 6s 37ms/step - loss: 0.0052 - accuracy: 0.2700 - cosine_similarity: 0.2554 - val_loss: 0.0072 - val_accuracy: 0.1628 - val_cosine_similarity: 0.1660

# Trains faster, better accuracy  => no need for attention


