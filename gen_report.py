import torch
import pickle
from utils import sample

class Gen_Report():
    def __init__(self):
        super(Gen_Report, self).__init__()
        with open("./db_vocab.pkl", "rb") as cache:
          self.db_vocab = pickle.load(cache)
          self.word_2_id = self.db_vocab["word_2_id"]
          self.id_2_word = self.db_vocab["id_2_word"]
  
    def generate(self, model, x, y, len_mark, label):
        batch_size = x.shape[0]
        gens = []
        for i in range(batch_size):
          x_prime = x[i]
          y_prime = y[i]
          #print("decoder input shape:", x)
          x_prime = x_prime.unsqueeze(0)
          y_prime = y_prime.unsqueeze(0).unsqueeze(2)
          gens.append(sample(model, x_prime, y_prime[:,0,:], label, None, steps=30, train=True))

        gen_texts = []
        for i in range(batch_size):
          gen_text = [self.id_2_word[k]  if  k != 2319 else '' for k in gens[i][1:15]]
        return gen_texts