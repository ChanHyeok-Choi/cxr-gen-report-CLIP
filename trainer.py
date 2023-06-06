import math
import logging

from tqdm import tqdm
import numpy as np
from torchtext.data.metrics import bleu_score
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from loss import Clip_loss
from utils import sample, save_loss_to_csv

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, word_2_id, id_2_word):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.word_2_id = word_2_id
        self.id_2_word = id_2_word

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def generate(self, model, x, y, len_mark, label):
        # print(x.shape, y.shape) # torch.Size([16, 224, 224]) torch.Size([16, 512, 1])
        batch_size = x.shape[0]
        ys = []
        gens = []
        for i in range(batch_size):
            x_prime = x[i]
            y_prime = y[i]
            label_prime = label[i]
            #print("decoder input shape:", x)
            x_prime = x_prime.unsqueeze(0)
            y_prime = y_prime.unsqueeze(0).unsqueeze(2)
            y_idx = [item.item() for sublist in y_prime[0] for item in sublist]
            gens.append(sample(model, x_prime, y_prime[:,0,:], label_prime, None, steps=30, train=True))
            ys.append(y_idx)

        y_texts = []
        gen_texts = []
        for i in range(batch_size):
            gen_texts.append([self.id_2_word[k]  if  k != 2319 else '' for k in gens[i][1:-2]])
            y_texts.append([self.id_2_word[k] if k != 2319 else '' for k in ys[i][1:-2]])
        return gen_texts, y_texts

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        with torch.no_grad():
            clip = Clip_loss()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                collate_fn = data.collate_fn)

            losses = []
            clip_losses = []
            tgts = []
            len_tgts = []
            preds = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, len_masks, labels) in pbar:
                # print(x.shape, y.shape) # torch.Size([16, 224, 224]) torch.Size([16, 512])

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                len_masks = len_masks.to(self.device)
                labels = labels.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss, pred = model(x, y, len_masks, labels)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    tgts.append(y)
                    #len_tgts.append(leny)
                    preds.append(pred)
                        
                    src_class, trg_class = self.generate(model, x, y, len_masks, labels)
                    clip_loss = 0.
                    batch_size = x.shape[0]
                    for i in range(batch_size):
                        src_text = ' '.join(src_class[i])
                        trg_text = ' '.join(trg_class[i])
                        src_class[i] = src_text
                        trg_class[i] = trg_text
                
                # print(x.shape)
                x_ = x.unsqueeze(1).repeat(1,3,1,1)
                # print(x_.shape)
                clip_loss = clip(x_, src_class, x_, trg_class)
                clip_losses.append(clip_loss)
                loss += clip_loss

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.requires_grad_(True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. clip loss {clip_loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                #logger.info("test loss: %f", test_loss)
                #return test_loss
           

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
                    tgts_list[-1] = [[self.id_2_word[x] for x in tgts_list[-1]]]

                for i in range(len(preds)):
                    try:
                        eos_ind = preds[i].index(2319)
                    except:
                        eos_ind = len(preds[i])-1

                    preds_list.append(preds[i][:eos_ind])
                    preds_list[-1] = [str(self.id_2_word[x]) for x in preds_list[-1]]

                assert(len(preds_list) == len(tgts_list))
                #print(preds_list[10][:60])
                #print(tgts_list[10][0][:60])
                test_bleu = bleu_score(preds_list, tgts_list, max_n=2, weights=[0,1])
            
                logger.info("test loss: %f \t bleu_score_2:%f", test_loss, test_bleu)

                clip_losses_float = [float(value) for value in clip_losses]
                save_loss_to_csv(epoch, losses, clip_losses_float, test_loss, test_bleu, '/content/drive/MyDrive/UNIST/2023_1/NLP/ChestXrayReportGen/cxr-report-generation/enc_dcd/loss_csv/test.csv')
                
                return test_loss, test_bleu


        best_loss = float('inf')
        best_bleu = float('-inf')

        self.tokens = 0 # counter used for learning rate decay
        
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss, test_bleu = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < 1.10*best_loss
            if self.config.ckpt_path is not None and good_model:
                if test_loss < best_loss:
                    best_loss = test_loss
                best_bleu = test_bleu
                self.save_checkpoint()
