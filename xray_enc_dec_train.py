import numpy as np
import pandas as pd
import torchvision
import torch
import pickle
from dataset import chestXRayDataset
from torch.utils.data import Dataset,  DataLoader
import logging
from utils import set_seed
from model import ImageEncoderReportDecoder, ImageEncoderReportDecoderConfig
from trainer import Trainer, TrainerConfig
from torchvision.models import efficientnet_b5

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

set_seed(42)
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

##################################
# load pretrained nets for image encoding
##################################
# Resnet 18  img_enc_width = img_enc_height = 224  / img_enc_out_shape = (512,1) / block_size = 512 / rgb = True
# img_enc = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
# img_enc.fc = torch.nn.Identity()

# Efficient U net
img_enc = efficientnet_b5(pretrained = True)
num_ftrs = img_enc.classifier[1].in_features
img_enc.classifier[1] = torch.nn.Linear(num_ftrs, 512)
img_enc.fc = torch.nn.Identity()

# Unet img_enc_width = img_enc_height = 256  / img_enc_out_shape = (256, 256) / block_size = 256 / rgb = True
#img_enc = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

# Directly feed the image into encoder. img_enc_width = img_enc_height = 224  / img_enc_out_shape = (224, 224) / block_size = 224 / rgb = False
#img_enc = torch.nn.Identity()

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
model = ImageEncoderReportDecoder(mconf, img_enc, img_enc_out_shape, rgb=rgb)
#print(model)
#model.load_state_dict(torch.load("./xray_model.pt"))

#################################
# set TrainerConfig and Trainer
###############################
tokens_per_epoch = len(train_dataset) * block_size
train_epochs = 500
tconf = TrainerConfig(max_epochs=train_epochs, batch_size=16, learning_rate=2e-3,
                      betas = (0.9, 0.95), weight_decay=0,
                      lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='xray_model_1.pth',
                      num_workers=8)
trainer = Trainer(model, train_dataset, val_dataset, tconf, word_2_id, id_2_word)
trainer.train()



