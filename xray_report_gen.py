import numpy as np
import pandas as pd
import torchvision
import torch
from dataset import chestXRayDataset
import pickle
from torch.utils.data import Dataset,  DataLoader
import logging
from utils import set_seed, sample
from model import ImageEncoderReportDecoder, ImageEncoderReportDecoderConfig
from trainer import Trainer, TrainerConfig
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from torchvision.models import efficientnet_b5

import evaluate
from ignite.metrics import Rouge

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

set_seed(42)


##################################
# load resnet18 for image encoding
##################################
img_enc = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
img_enc.fc = torch.nn.Identity()
# img_enc = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
# img_enc = torch.nn.Identity()

##################################
# load efficient U net for image encoding
##################################
# img_enc = efficientnet_b5(pretrained = True)
# num_ftrs = img_enc.classifier[1].in_features
# img_enc.classifier[1] = torch.nn.Linear(num_ftrs, 512)
# img_enc.fc = torch.nn.Identity()

print(img_enc)
img_enc_width =  224 #256
img_enc_height = 224 #256
img_enc_out_shape = (512,1) #(256, 256) #(224, 224) # (512,1) # (256, 256)
block_size = img_enc_out_shape[0]
rgb = True


#################################
# load vocabulary and dataframes
#################################
data_path = "/content/drive/MyDrive/UNIST/2023_1/NLP/ChestXrayReportGen/dataset/IUXray"

with open("./db_vocab.pkl", "rb") as cache:
    db_vocab = pickle.load(cache)
    word_2_id = db_vocab["word_2_id"]
    id_2_word = db_vocab["id_2_word"]

vocab_size = len(word_2_id)
assert(len(id_2_word) == len(word_2_id))
print("vocabulary size:", len(id_2_word))


with open("./db_datasets.pkl", "rb") as cache:
    db_database = pickle.load(cache)
    train_df = db_database["train_df"]
    val_df = db_database["val_df"]



##################################
# generate train/validation sets
#################################
train_dataset = chestXRayDataset(train_df, data_path, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word)
val_dataset = chestXRayDataset(val_df, data_path, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word)
print(f'There are {len(train_dataset) :,} samples for training, and {len(val_dataset) :,} samples for validation testing')

####################################
# create the encoder/decoder model
###################################
mconf = ImageEncoderReportDecoderConfig(vocab_size, block_size, n_embd=img_enc_width)
model = ImageEncoderReportDecoder(mconf, img_enc, img_enc_out_shape, rgb = rgb)
model.load_state_dict(torch.load("./xray_model_original.pth")) # xray_model_original.pth, xray_model_original_CLIP_24.pth, xray_model_efficientUNet.pth, xray_model_efficientUNet_CLIP_27.pth
#model.img_enc = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

tgts = []
preds = []


print("len val dataset:", len(val_dataset))
pbar = tqdm(enumerate(val_dataset)) 

for it, (x, y, _, label) in pbar:
    # print(x.shape, y.shape) # torch.Size([224, 224]) torch.Size([512])
    if it >= len(val_dataset)-1:
        break
    #print("decoder input shape:", x)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(2)
    gen = sample(model, x, y[:,0,:], label, None, steps=30)

    y = [item.item() for sublist in y[0] for item in sublist]
    tgts.append(torch.IntTensor(y))
    preds.append(torch.IntTensor(gen))
    # print("y: ", y)
    # print("gen: ", gen)

    gen_text = [id_2_word[k]  if  k != 2319 else '' for k in gen[1:15]]
    print("\npredicted:", gen_text)
    # y = y.tolist()[0]
    # y = [item for sublist in y for item in sublist]
    y_text = [id_2_word[k] if k != 2319 else '' for k in y[1:15]]
    print("actual:", y_text)
    pbar.set_description(f"{it}/{len(val_dataset)}")



tgts = torch.vstack(tgts).cpu().numpy().tolist()
preds = torch.vstack(preds).cpu().numpy().tolist()
tgts_list = []
preds_list = []
for i in range(len(tgts)):
    try:
        eos_ind = tgts[i].index(2319)
    except:
        eos_ind = len(tgts[i])-1

    tgts_list.append(tgts[i][:eos_ind])
    tgts_list[-1] = [[id_2_word[x] for x in tgts_list[-1]]]

for i in range(len(preds)):
    try:
        eos_ind = preds[i].index(2319)
    except:
        eos_ind = len(preds[i])-1
    preds_list.append(preds[i][:eos_ind])
    preds_list[-1] = [str(id_2_word[x]) for x in preds_list[-1]]

assert(len(preds_list) == len(tgts_list))
#print(preds_list[10])
#print(tgts_list[10])
test_bleu_1 = bleu_score(preds_list, tgts_list, max_n=1, weights=[1])   
test_bleu_2 = bleu_score(preds_list, tgts_list, max_n=2, weights=[0,1])
test_bleu_3 = bleu_score(preds_list, tgts_list, max_n=3, weights=[0,0,1])
test_bleu_4 = bleu_score(preds_list, tgts_list, max_n=4, weights=[0,0,0,1])
test_bleu_5 = bleu_score(preds_list, tgts_list, max_n=5, weights=[0,0,0,0,1])
print(f"Bleu Scores:1:{test_bleu_1} \t 2:{test_bleu_2} \t 3:{test_bleu_3} \t 4:{test_bleu_4} \t 5:{test_bleu_5}")

#######################
# Rouge Score
#######################
m = Rouge(variants=["L", 2], multiref="best")
m.update((preds_list, tgts_list))
print(m.compute())

#######################
# METEOR Score
#######################
print(preds_list)
print(tgts_list)
preds_reports = []
tgts_reports = []

for pred in preds_list:
    predicted = ' '.join(pred)
    preds_reports.append(predicted)
for trg in tgts_list:
    trg = [*trg[0]]
    target = ' '.join(trg)
    tgts_reports.append(target)
meteor = evaluate.load('meteor')
print(meteor.compute(predictions=preds_reports, references=tgts_reports))