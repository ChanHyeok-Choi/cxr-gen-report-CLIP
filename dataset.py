import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch
import torchvision
from torchvision import transforms


TRAIN_SIZE = 0.8
SEQUENCE_LEN = 1500

def generate_vocabulary(df):

    df = df.fillna('')
    
    words = []
    for (report, finding) in zip(df.impression, df.findings):
        rp = re.findall(r"[\w']+|[.,!?;]", report)
        fg = re.findall(r"[\w']+|[.,!?;]", finding)
        words.append(rp)
        words.append(fg)
        #words.append(str.split(report))

    vocab = [x for sublist in words for x in sublist]

    vocab = sorted(np.unique(vocab))

    word_2_id = {}
    id_2_word = {}
    for ind, word in enumerate(vocab):
        word_2_id[str(word)] = ind
        id_2_word[ind] = word

    vocab_size = len(vocab)
    word_2_id['<eos>'] = vocab_size
    id_2_word[vocab_size] = '<eos>'
    word_2_id['<start>'] = vocab_size+1
    id_2_word[vocab_size+1] = '<start>'

    #print(word_2_id)
    #print(id_2_word)
    print("len vocab:", len(vocab))
    return word_2_id, id_2_word



def get_train_val_df(df):
  """
    Separates the dataframe into training and validation sets. Splits by subject id.
  """
  train_split = TRAIN_SIZE
  ids = df.uid.unique()
  np.random.seed(1)
  train_uids = np.random.choice(ids, size = int(len(ids)*train_split), replace = False)

  df['in_train'] = None
  df['in_train'] = df["uid"].apply(lambda x: x in train_uids)
  train_df = df[df['in_train'] == True]
  val_df = df[df['in_train'] == False]

  return train_df, val_df

    
class chestXRayDataset(Dataset):
    def __init__(self, df, img_dir, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word):
        
        self.block_size = block_size
        self.img_enc_width = img_enc_width
        self.img_enc_height = img_enc_height

        df = df.reset_index()
        self.img_labels = df[['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Lesion', \
                              'Lung Opacity','Edema','Consolidation','Pneumonia','Atelectasis',\
                              'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']]
        self.word_2_id = word_2_id
        self.id_2_word = id_2_word

        self.img_labels = self.img_labels.fillna(2)
        df.impression = df.impression.fillna('')
        df.findings = df.findings.fillna('')


        self.img_labels = self.img_labels + 1
        self.num_labels = len(self.img_labels.columns)
        self.img_labels = self.img_labels.to_numpy()

        self.findings = df['findings'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]", x))
        self.findings = self.findings.apply(lambda x: [self.word_2_id[str(word)] for word in x])
        self.findings_len = self.findings.apply(lambda x: len(x)+2)
        self.findings = self.findings.apply(lambda x: np.pad(x, (1,block_size-1-len(x)), constant_values=(word_2_id['<start>'],word_2_id['<eos>'])))


        self.impression = df['impression'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]", x))
        self.impression = self.impression.apply(lambda x: [self.word_2_id[str(word)] for word in x])
        self.impression_len = self.impression.apply(lambda x: len(x)+2)
        self.impression = self.impression.apply(lambda x: np.pad(x, (1,block_size-1-len(x)), constant_values=(word_2_id['<start>'],word_2_id['<eos>'])))

        self.img_files = df['filename'].apply(lambda x: os.path.join(img_dir+"/images/images_normalized/", x))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        with Image.open(self.img_files[idx]) as image:
            image.load()
        image = image.resize((self.img_enc_height, self.img_enc_height))
        image = np.asarray(image)
        image2 = np.array(image , dtype=float)
        image = image2/255.0
        
        #rn = np.random.uniform()
        #if rn < 0.30:
        #    image = np.array(np.flipud(image))
        #elif rn >= 0.30 and rn < 0.60:
        #    image = np.array(np.fliplr(image))

        m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
            ])
        image  = preprocess(image).squeeze(0).type(torch.FloatTensor)
        
        label = self.img_labels[idx]

        #rn = np.random.uniform()
        #if rn < 0.5:
        report = self.findings[idx]
        len_mask = [False if i < self.findings_len[idx] else True for i in range(self.block_size)]
        #else:
        #    report = self.impression[idx]
        #    len_mask = [False if i < self.impression_len[idx] else True for i in range(self.block_size)]

        labels = self.img_labels[idx].tolist()


        return image, torch.LongTensor(report), torch.BoolTensor(len_mask), torch.IntTensor(labels)
    
    def collate_fn(self, samples):
        images, reports, len_masks, labels = [], [], [], []
        for image, report, len_mask, label in samples:
            images.append(image)
            reports.append(report)
            len_masks.append(len_mask)
            labels.append(label)

        
        reports = pad_sequence(reports, batch_first=True, padding_value=self.word_2_id['<eos>'])
        images = pad_sequence(images, batch_first=True)
        len_masks = torch.vstack(len_masks)
        labels = torch.vstack(labels)
        return images, reports, len_masks, labels

