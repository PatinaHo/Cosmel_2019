#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import re
import csv
import time
import math
import json
import itertools
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

device = torch.device("cuda:0") # Uncomment this to run on GPU
WORD_DIM = 300

def main():
    # Loading pretrained embedding, create embedding layer
    embedding = load_embedding('/share/home/patina/styleme/data/wordemb/ASBC_CNA_WIKI_u8.word_sg_w15_n25_300d_min5_i15.vec')
    vocab = load_vocab('/share/home/patina/styleme/data/wordemb/ASBC_CNA_WIKI_u8.word_sg_w15_n25_300d_min5_i15.vocab')
    # Loading, preprocessing and embedding input texts
    ptt_makeup_sent = load_text('/share/home/patina/styleme/data/ptt_makeup_ws.tsv', max_sent=15000)
    ptt_gossip_sent = load_text('/share/home/patina/styleme/data/ptt_gossip_ws.tsv', max_sent=30000)
    ptt_makeup_sent = cut_sent(sent_list=ptt_makeup_sent, limit_len=40)
    ptt_gossip_sent = cut_sent(sent_list=ptt_gossip_sent, limit_len=40)
    all_ptt_sent = ptt_makeup_sent + ptt_gossip_sent
    all_ptt_sent_idx = [token_to_index(sent, vocab) for sent in all_ptt_sent]
    all_ans = np.append(np.ones(len(ptt_makeup_sent)), np.zeros(len(ptt_gossip_sent)))

    # Splitting data into training, validate, test set
    sent_train, sent_test, ans_train, ans_test = train_test_split(all_ptt_sent_idx, all_ans, test_size=0.3, random_state=1)
    sent_train, sent_val, ans_train, ans_val = train_test_split(sent_train, ans_train, test_size=0.2, random_state=1)

    best_model = train_and_eval(num_model_to_train=5, training_set=sent_train, training_ans=ans_train, eval_set=sent_val, eval_ans=ans_val, embedding=embedding)
    torch.save(best_model.state_dict(), './model1.pt')
    


def load_embedding(w2v_filepath):
    """
    Load pretrained word2vec file, create and return an embedding layer.
    """
    print('Start loading Word2vec model...')
    tStart = time.time()
    path = get_tmpfile('/share/home/patina/styleme/data/wordemb/ASBC_CNA_WIKI_u8.word_sg_w15_n25_300d_min5_i15.vec')
    model = KeyedVectors.load_word2vec_format(path, unicode_errors='ignore')
    weights = torch.FloatTensor(model.vectors)
    tEnd = time.time()
    print('Word2vec model loaded, costs %f sec.' % (tEnd-tStart))
    print()
    embedding = nn.Embedding.from_pretrained(weights)
    return embedding

def load_vocab(vocab_jsonfile):
    print('Start loading Vocab...')
    with open(vocab_jsonfile) as vocab_jsonfile:
        vocab = json.load(vocab_jsonfile)
    print('Vocab Loaded.')
    print()
    return vocab

def load_text(ptt_tsvfile, *, max_sent):
    """ Return a numpy array of all sentences.
    """
    unclear_sents_set = set()
    sents_set = set()
    sents_list = []
    with open(ptt_tsvfile, newline='') as tsvfile:
        rows = csv.reader(tsvfile, delimiter='\t')
        for row_id, row in enumerate(rows):    # (sent, count) in row
            if row_id < max_sent:
                    unclear_sents_set.add(row[0].strip())
            else:
                break
    for unclear_sent in unclear_sents_set:
        clearsent = re.split('\t|\n', unclear_sent)
        for sent in clearsent:
            sents_set.add(sent)
    sents_list = [sent.split() for sent in sents_set]
    return sents_list

def token_to_index(word_list, vocab):
    idx_line = []
    for word in word_list:
        if word in vocab:
            idx_line.append(vocab[word])
        else:
            idx_line.append(0)
    return idx_line

def idxline_to_tensor(idx_line):
    tensor = torch.zeros(len(idx_line), 1, WORD_DIM)
    for li, letter in enumerate(idx_line):
        tensor[li][0] = embedding(torch.LongTensor([idx_line[li]]))
    return tensor

def cut_sent(sent_list, limit_len):
    def chunks(sent_list, limit_len):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(sent_list), limit_len):
            yield sent_list[i:i+limit_len]
            
    list_after_cut = []
    for sent in sent_list:
        if(len(sent) > limit_len):
            for fragment in chunks(sent, limit_len):
                list_after_cut.append(fragment)
        else:
            list_after_cut.append(sent)
    return list_after_cut

def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def inputVar(input_batch, embedding):
    """ Returns padded input sequence tensor and lengths.
    """
    lengths = torch.tensor([len(indexes) for indexes in input_batch])
    padList = zeroPadding(input_batch)
    padVar = torch.stack([embedding(torch.LongTensor(item)) for item in padList], dim=0)
    return padVar, lengths

def batch_grouping(batch_size, input_tensors, outputs, embedding, drop_last=True):
    assert len(input_tensors) == len(outputs), 'LEN NOT MATCH! YOU IDIOT!!!'
    
    num_data = len(input_tensors)
    batches = []
    idx_list = list(range(0, num_data, batch_size))
    
    if not drop_last and idx_list[-1] != num_data: idx_list.append(num_data)
    
    for idx0, idx1 in zip(idx_list[:-1], idx_list[1:]):
        input_batch = input_tensors[idx0:idx1]
        output_batch = outputs[idx0:idx1]
            
        inp, input_lengths = inputVar(input_batch, embedding=embedding)
        out = torch.LongTensor(output_batch)
        
        assert inp.shape[1] == input_lengths.shape[0] == out.shape[0]
        
        batch_reidx = sorted(range(len(input_lengths)), key=lambda k: input_lengths[k], reverse=True)
        
        inp           = inp[:, batch_reidx, :]
        input_lengths = input_lengths[batch_reidx]
        out           = out[batch_reidx]
        
        batches.append([inp, input_lengths, out])
            
    return batches

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Define Model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input_tensor):
        # input_tensor # (seq_len, batch_size, WORD_DIM)
        hidden_all, _ = self.lstm(input_tensor) # (seq_len, batch_size, WORD_DIM)
        hidden = hidden_all[-1, :, :] # (batch_size, WORD_DIM)
        
        hidden = torch.sigmoid(self.linear1(hidden))
        hidden = torch.sigmoid(self.linear2(hidden))
        output = self.linear3(hidden)
        
        return output

    def loss(self, output, category_tensor):
        loss = torch.nn.functional.cross_entropy(output, category_tensor)
        
        return loss


def train_model(n_hidden=128, n_categories=2, print_every=500, plot_every=500, *, training_set, training_ans, batch_size, embedding, epochs):
    batches = batch_grouping(batch_size=batch_size, input_tensors=training_set, outputs=training_ans, embedding=embedding)
    model = MyModel(WORD_DIM, n_hidden, n_categories)
    model.cuda()
    optimizer = optim.Adam(model.parameters())

    model.zero_grad()
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    for epoch in range(epochs):
        for batch_id, batch in enumerate(batches):
            input_tensors, input_lengths, ans = batch
            output = model(input_tensors.cuda())
            loss = model.loss(output.cuda(), ans.cuda())
            current_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#             # Print iter number, loss, name and guess
#             if batch_id % print_every == print_every-1:
#                 print('[%d %d] %d%% (%s) Loss: %.4f' % (epoch+1, batch_id+1, batch_id/len(batches)*100, timeSince(start), loss))

            # Add current loss avg to list of losses
            if batch_id % plot_every == plot_every-1:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    print("Trainning has been done. Loss: %.4f" % (loss))
    
    return model, all_losses


def eval_model(n_hidden=128, n_categories=2, print_every=5, *, model, eval_set, eval_ans, batch_size, embedding):
    total_loss = 0
    model.eval()
    batches = batch_grouping(batch_size=batch_size, input_tensors=eval_set, outputs=eval_ans, embedding=embedding, drop_last=False)
    
    start = time.time()
    prediction_all = np.array([])
    ans_all        = np.array([])
    for batch_id, batch in enumerate(batches):
        
        input_tensors, input_lengths, ans = batch
        prediction = model(input_tensors.cuda())
        loss = model.loss(prediction.cuda(), ans.cuda())
        total_loss += loss
        
        # To calculate accuracy
        prediction_np = prediction.cpu().data.numpy()        
        prediction_np = np.argmax(prediction_np, axis=1)
        ans_np        = ans.cpu().data.numpy()
        
        prediction_all = np.append(prediction_all, prediction_np)
        ans_all = np.append(ans_all, ans_np)
    
    acc = accuracy_score(ans_all, prediction_all)
    return acc, total_loss.item()/len(batches)
    
def train_and_eval(num_model_to_train, training_set, training_ans, eval_set, eval_ans, embedding):
    models = []
    eval_losses = []
    
    print("Start training ...")
    for i in range(num_model_to_train):
        model, loss = train_model(training_set=training_set, training_ans=training_ans, batch_size=128, embedding=embedding, epochs=30)
        print("Model%d trained. Loss: %4f" % (i, loss[-1]))
        models.append(model)
    
    print()
    print("Start evaluating ...")
    for j in range(len(models)):
        acc, loss = eval_model(model=models[j], eval_set=eval_set, eval_ans=eval_ans, batch_size=128, embedding=embedding)
        print("Model%d evaluated. Loss: %4f; accuracy: %4f" % (j, loss, acc))
        eval_losses.append(loss)
        
    best_model_idx = eval_losses.index(min(eval_losses))
    
    return models[best_model_idx]


if __name__ == "__main__":
    main()