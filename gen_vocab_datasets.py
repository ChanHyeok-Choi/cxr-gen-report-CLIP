import numpy as np
import pandas as pd
import pickle
from dataset import get_train_val_df, generate_vocabulary
import logging
from utils import set_seed

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

set_seed(42)
data_path = "/home/desin/CS224N/data/chestXray/"


#################################
# generate vocabulary
#################################
df = pd.read_csv(data_path+"uid_report_projection_label.csv")
word_2_id, id_2_word = generate_vocabulary(df)
vocab_size = len(word_2_id)
assert(len(id_2_word) == len(word_2_id))
print("vocabulary size:", len(id_2_word))



##################################
# generate train/validation sets
#################################
df = pd.read_csv(data_path+"uid_report_projection_label.csv")
df = df[df['filename'].notna()]
train_df, val_df = get_train_val_df(df)
print(f'There are {len(train_df) :,} samples for training, and {len(val_df) :,} samples for validation testing')

db_vocab = {"word_2_id":word_2_id, "id_2_word":id_2_word}
with open("./db_vocab.pkl", "wb") as cache:
    pickle.dump(db_vocab, cache)


db_dataset = {"train_df":train_df, "val_df":val_df}
with open("./db_datasets.pkl", "wb") as cache:
    pickle.dump(db_dataset, cache)







