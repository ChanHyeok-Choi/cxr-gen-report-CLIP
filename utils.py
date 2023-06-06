import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import csv
import ast
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def sample(model, x, y0, label, len_mask, steps, train=False):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    if train:
        with torch.enable_grad():
            block_size = model.module.get_block_size() # train
            generated = []

            #print("x >>>>>>>>>>:", x)
            y1 = y0
            reps = model.module.representation(x, label) # train
            #print(reps)

            for k in range(steps):
                
                #print("****************", y0)
                logits, _, pred = model.module.decode(reps, y0, None) # train
                #logits, _, pred = model(x, y0, None, label)
                
                #for j in range(len(y0[0])):
                #    y2 = torch.argmax(logits[:, j, :], dim=1)
                #    print("&&&&&&&&&&&&&&&&", y2)


                logits = logits[:, len(y0)-1, :]

                
                #logits[:, y1] = float('-inf')
                y1 = torch.argmax(logits,dim=1).unsqueeze(0)
                #print(y1.item())
                generated.append(y1.item())
                #print("y0 shape:", y0.shape)
                #print("y1 shape:", y1.shape)
                y0 = torch.cat((y0, y1), dim=1)
                
            return generated
    else:
        with torch.no_grad():
            block_size = model.get_block_size() # generate
            model.eval()
            generated = []

            #print("x >>>>>>>>>>:", x)
            y1 = y0
            reps = model.representation(x, label) # generate
            #print(reps)

            for k in range(steps):
                
                #print("****************", y0)
                logits, _, pred = model.decode(reps, y0, None) # generate
                #logits, _, pred = model(x, y0, None, label)
                
                #for j in range(len(y0[0])):
                #    y2 = torch.argmax(logits[:, j, :], dim=1)
                #    print("&&&&&&&&&&&&&&&&", y2)


                logits = logits[:, len(y0)-1, :]

                
                #logits[:, y1] = float('-inf')
                y1 = torch.argmax(logits,dim=1).unsqueeze(0)
                #print(y1.item())
                generated.append(y1.item())
                #print("y0 shape:", y0.shape)
                #print("y1 shape:", y1.shape)
                y0 = torch.cat((y0, y1), dim=1)
                
            return 

def save_loss_to_csv(epoch, train_losses, clip_losses, test_loss, bleu_score, filename='/content/drive/MyDrive/UNIST/2023_1/NLP/ChestXrayReportGen/cxr-report-generation/enc_dcd/loss_csv/test.csv'):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['epoch', 'train_loss', 'clip_loss', 'test_loss', 'bleu_score'])
        writer.writerow((epoch, train_losses, clip_losses, test_loss, bleu_score))

def read_loss_from_csv(filename='/content/drive/MyDrive/UNIST/2023_1/NLP/ChestXrayReportGen/cxr-report-generation/enc_dcd/loss_csv/test.csv'):
    epoch_list = []
    train_loss_list = []
    clip_loss_list = []
    test_loss_list = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        epoch_index = header.index('epoch')
        train_loss_index = header.index('train_loss')
        clip_loss_index = header.index('clip_loss')
        test_loss_index = header.index('test_loss')
        for row in reader:
            epoch = int(row[epoch_index])
            train_loss = ast.literal_eval(row[train_loss_index])
            clip_loss = ast.literal_eval(row[clip_loss_index])
            test_loss = ast.literal_eval(row[test_loss_index])
            epoch_list.append(epoch)
            train_loss_list.append(train_loss)
            clip_loss_list.append(clip_loss)
            test_loss_list.append(test_loss)
    return epoch_list, train_loss_list, clip_loss_list, test_loss_list

def plot_mean_loss(epoch_list, train_losses, clip_losses, test_loss, title: str, plot_path='/content/drive/MyDrive/UNIST/2023_1/NLP/ChestXrayReportGen/cxr-report-generation/enc_dcd/plot'):
    epoch = epoch_list
    train_mean_loss = np.mean(train_losses, axis=1)
    clip_mean_loss = np.mean(clip_losses, axis=1)

    plt.figure()
    plt.plot(epoch, train_mean_loss, label='CE_mean_loss')
    plt.plot(epoch, clip_mean_loss, label='clip_mean_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(plot_path + '/' + title + '.png')
    plt.show()

def plot_iteration_loss(loss_list, loss_name:str, title: str, plot_path='/content/drive/MyDrive/UNIST/2023_1/NLP/ChestXrayReportGen/cxr-report-generation/enc_dcd/plot'):
    iters = [i for i in range(len(loss_list))]

    plt.figure()
    plt.plot(iters, loss_list)
    plt.xlabel('Iteration')
    plt.ylabel(loss_name)
    plt.title(title)
    plt.savefig(plot_path + '/' + title +'.png')
    plt.show()

# epoch_list, train_loss_list, clip_loss_list, test_loss_list = read_loss_from_csv()
# plot_mean_loss(epoch_list, train_loss_list, clip_loss_list, test_loss_list, 'ResNet + CLIP Loss')