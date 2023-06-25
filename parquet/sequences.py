import pandas as pd
import numpy as np
import os

class Sequence: 
    def __init__(self, df, text):
        # DataFrame
        self.df = df
        # Sequence Text
        self.text = text



"""
        Load a parquet file containing sequences.
        Paths to the parquet file and metadata file
        are expected.

        String -> String -> Array[Sequence]
"""
def load_parquet(filepath, supplemental_metadata_path): 
    seqs = []

    csv = pd.read_csv(supplemental_metadata_path)

    # Extract file id from file name Ex. 134343541.parquet -> 134343541
    file_id = os.path.basename(filepath).split('.')[0]

    landmark_df = pd.read_parquet(filepath, 'pyarrow')

    for seq in list(set(landmark_df.index.tolist())):
        text = csv.loc[csv['sequence_id'] == seq]['phrase'].values[0]

        seq_df = landmark_df.loc[seq]
        seq_df_filtered = pd.concat( [\
            seq_df.loc[:, 'x_left_hand_0':'x_left_hand_20'], \
            seq_df.loc[:, 'y_left_hand_0':'y_left_hand_20'], \
            seq_df.loc[:, 'x_right_hand_0':'x_right_hand_20'], \
            seq_df.loc[:, 'y_right_hand_0':'y_right_hand_20']], axis=1)

        seqs.append(Sequence(seq_df_filtered, text))

    return seqs

def load_parquet_as_np(filepath, supplemental_metadata_path): 
    csv = pd.read_csv(supplemental_metadata_path)

    # Extract file id from file name Ex. 134343541.parquet -> 134343541
    file_id = os.path.basename(filepath).split('.')[0]

    landmark_df = pd.read_parquet(filepath, 'pyarrow').drop('frame', axis='columns')
    arr = []
    for seq in set(landmark_df.index):
        arr.append(csv.loc[csv['sequence_id'] == seq]['phrase'].values[0])
    return np.array(list(landmark_df.groupby(['sequence_id']).apply(lambda x:x.to_numpy()))), np.array(arr)

