import json
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
