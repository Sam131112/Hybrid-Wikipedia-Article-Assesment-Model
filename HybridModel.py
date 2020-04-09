import re
import torchtext
import torch
import spacy
from tqdm import tqdm
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
from torchtext.vocab import Vectors
import torch
from sklearn.model_selection import train_test_split
NLP = spacy.load('en')
MAX_CHARS = 40000
LEARNING_RATE = 1e-4
Batch_Size = 32
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def prepipeline(x):
    x = x.split()
    x = [float(_x) for _x in x]
    return x

def postpipeline(x):
    return x


class SequenceModel(torch.nn.Module):
        def __init__(self,vocab_size,ext_feats_sz,embed_size,hidden_size,Field_obj,out_class):
            super(SequenceModel,self).__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.embed_size = embed_size
            self.out_class = out_class
            self.ext_feats_size = ext_feats_sz
            self.embedding_layer = torch.nn.Embedding(Field_obj.vocab.vectors.shape[0],\
                    Field_obj.vocab.vectors.shape[1])
            self.embedding_layer.weight.data.copy_(Field_obj.vocab.vectors)
            self.embedding_layer.requires_grad = True
            self.BiLstm = torch.nn.LSTM(self.embed_size,self.hidden_size,\
                    dropout=0.4,num_layers=2,bidirectional=True)
            self.fc1= torch.nn.Linear(self.hidden_size*2,128)
            self.fc2 = torch.nn.Linear(128+self.ext_feats_size,64)
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            self.relu = torch.nn.ReLU()
            self.drop = torch.nn.Dropout(0.5)
            self.batchnorm128 = torch.nn.BatchNorm1d(128)
            self.batchnorm64 = torch.nn.BatchNorm1d(64)
            self.fc2 = torch.nn.Linear(128,64)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            self.fc3 = torch.nn.Linear(64,self.out_class)
            torch.nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self,seq,seq_len,feats):
            encoded = self.embedding_layer(seq)
            packed_padded_seq = torch.nn.utils.rnn.pack_padded_sequence(encoded,seq_len)
            packed_output , hidden = self.BiLstm(packed_padded_seq)
            output,out_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
            output,_ = torch.max(output,axis=0) # MaxPooling along sequences
            out_fc1 = self.fc1(output)

            

            return predict

def spacy_tokenizer(x):
    my_tokenizer = spacy.load('en')
    return [tok.text for tok in my_tokenizer(x)]

def new_tokenizer(comment):
    comment = re.sub(r"[\*\"\n\\\+\-\/\=\(\):\[\]\|\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [x.text for x in NLP.tokenizer(comment) if x.text != " "]

def basic_tokenizer(x):
    return x.split(" ")


def prepare_data():
    TEXT = torchtext.data.Field(sequential=True,lower=True,\
            tokenize = new_tokenizer,stop_words=STOP_WORDS,fix_length=256,\
            is_target=False,include_lengths=True)
    LABEL = torchtext.data.Field(sequential=False,use_vocab=False,is_target=True)
    FEATS = torchtext.data.RawField(preprocessing=torchtext.data.Pipeline(prepipeline),\
            postprocessing=torchtext.data.Pipeline(postpipeline),is_target=False)
    pretrained_vector = Vectors("glove.6B.50d.txt")
    train_fields = [("Clean_Text",TEXT),("label",LABEL),("Feats",FEATS)]
    train = torchtext.data.TabularDataset('train.csv',format='csv',\
            skip_header=True,fields=train_fields)
    val = torchtext.data.TabularDataset('validation.csv',format='csv',\
            skip_header=True,fields=train_fields)
    test_fields = [("Clean_Text",TEXT),('label',LABEL),,("Feats",FEATS)]
    test = torchtext.data.TabularDataset('test.csv',format='csv',skip_header=True,\
                fields=test_fields)

    TEXT.build_vocab(train,max_size=75000,min_freq=20,\
            vectors=pretrained_vector,unk_init=torch.Tensor.uniform_)
    train_iter = torchtext.data.BucketIterator(train,device=device,\
            batch_size=Batch_Size,sort_key=lambda x:len(x.Clean_Text),\
            sort_within_batch=True)

    val_iter = torchtext.data.BucketIterator(val,device=device,\
            batch_size=Batch_Size,sort_key=lambda x:len(x.Clean_Text),\
            sort_within_batch=True,train=False)
    test_iter = torchtext.data.BucketIterator(test,device=device,\
            batch_size=Batch_Size,sort_within_batch=True,train=False,\
            sort_key=lambda x:len(x.Clean_Text))

    return (train_iter,val_iter,test_iter,TEXT)

def train_engine(model,train_iter,val_iter,epochs):

    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    validation_best = 0
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,\
            max_lr=0.1,steps_per_epoch=len(train_iter),epochs=epochs)
    for epoch in range(epochs):
        print("Trainning Epoch ",epoch)
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            #print("Iteration ",batch)
            seq , seq_len = batch.Clean_Text
            y = batch.label
            feats = torch.tensor(batch.Feats,dtype=torch.float32)
            seq.to(device)
            seq_len.to(device)
            y.to(device)
            feats.to(device)
            #print(seq,seq_len,y)
            pred = model(seq,seq_len,feats)
            loss = criteria(pred,y)
            train_loss = train_loss + loss.item()
            _,predictions = torch.max(pred,axis=1)
            train_accuracy = train_accuracy + torch.eq(predictions,y).sum()/(len(y)*1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # VALIDATION
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            model.eval()
            for batch in tqdm(val_iter):
                seq , seq_len = batch.Clean_Text
                y = batch.label
                feats = torch.tensor(batch.Feats,dtype=torch.float32)
                feats.to(device)
                seq.to(device)
                seq_len.to(device)
                y.to(device)
                pred = model(seq,seq_len,feats)
                loss = criteria(pred,y)
                val_loss = val_loss + loss.item()
                _,predictions = torch.max(pred,axis=1)
                val_accuracy = val_accuracy + torch.eq(predictions,y).sum()/(len(y)*1.0)


        val_epoch_accuracy = (val_accuracy/(len(val_iter)*1.0)).item()
        if val_epoch_accuracy >= validation_best:
            torch.save({
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss,},'saved_model.pth')
            validation_best = val_epoch_accuracy
        print("Training Loss , Training Accuracy ",(train_loss/(len(train_iter)*1.0)),\
                (train_accuracy/(len(train_iter)*1.0)).item())
        print("Validation Loss , Validation Accuracy ",(val_loss/(len(val_iter)*1.0)),\
                (val_accuracy/(len(val_iter)*1.0)).item())


def test_engine(model,test_iter):
        with torch.no_grad():
            test_loss = 0.0
            test_accuracy = 0.0
            model.eval()
            for batch in tqdm(test_iter):
                seq , seq_len = batch.Clean_Text
                y = batch.label
                feats = torch.tensor(batch.Feats,dtype=torch.float32)
                feats.to(device)
                seq.to(device)
                seq_len.to(device)
                y.to(device)
                pred = model(seq,seq_len,feats)
                _,predictions = torch.max(pred,axis=1)
                test_accuracy = test_accuracy + torch.eq(predictions,y).sum()/(len(y)*1.0)


        print("Final Test Accuracy ",test_accuracy/(len(test_iter)*1.0))

def main():
    df = pd.read_csv("train.csv")
    train_iter ,val_iter , test_iter , TXT  = prepare_data()
    print("Data Preprocessed")
    hidden = 256
    vocab , embed  = TXT.vocab.vectors.shape[0],TXT.vocab.vectors.shape[1]
    class_sz = len(df["label"].unique())
    model = SequenceModel(vocab,embed,hidden,TXT,class_sz)
    model.to(device)
    print("Start Training",TXT.vocab.vectors.shape)
    train_engine(model,train_iter,val_iter,25)
    model_best = SequenceModel(vocab,embed,hidden,TXT,class_sz)
    checkpoint = torch.load("saved_model.pth")
    model_best.load_state_dict(checkpoint["model_state_dict"])
    model_best.to(device)
    test_engine(model_best,test_iter)


if __name__ == "__main__":
    main()
