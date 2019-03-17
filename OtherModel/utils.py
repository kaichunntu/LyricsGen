import os
import pickle as pk
import numpy as np
from collections import defaultdict


def exist_or_mkdir(root,path):
    path = os.path.join(root , path)
    if(not os.path.exists(path)):
        os.mkdir(path)
    print("Path : '{}'".format(path))
    return path


def transform_orig(x , min_count = 3,max_len = [50,30],path=None):
    en_len , de_len = max_len
    print("Min Count :" , min_count)
    print("Max Length :" , max_len)
    word_freq = defaultdict(lambda: 0)
    for xx in x:
        for sr in xx:
            # Record frequency of word
            for s in sr:
                word_freq[s] += 1
    if(path):
        print("Use pre-trained tokenizer")
        idx2word,word2idx = pk.load(open(path,"rb"))["orig_word"]
    else:
        idx2word = ["NULL" , "OOV" , "SOS" , "EOS"]
        idx2word.extend([ k for k,v in word_freq.items() if(v >= min_count and (k not in ["SOS" , "EOS"]) )])
        word2idx = dict([(k,i) for i,k in enumerate(idx2word)])
    print("Word Count :" , len(idx2word))
    
    idx_in_sen = []
    mask_in = []
    idx_out_sen = []
    mask_out = []
    remain_idx = []
    length_in = []
    length_out = []
    
    for c , (s1 , s2) in enumerate(zip(*x)):
        ## remove SOS
        s1 = s1[1:]
        
        if(len(s1) > en_len or len(s2) > de_len ):
            continue
        remain_idx.append(c)
        length_in.append(len(s1))
        ## For dynamic rnn, using post padding
        mask_in.append(np.zeros([en_len]))
        mask_in[-1][0:len(s1)] += 1
        tmp_in_sen = []
        for s in s1:
            try:
                tmp_in_sen.append(word2idx[s])
            except:
                tmp_in_sen.append(1)
        tmp_in_sen.extend([0]*(en_len - len(s1)))
        idx_in_sen.append(tmp_in_sen)
        
        tmp_out_sen = []
        mask_out.append(np.zeros([de_len]))
        mask_out[-1][0:len(s2)] += 1
        for s in s2[1::]:
            try:
                tmp_out_sen.append(word2idx[s])
            except:
                tmp_out_sen.append(1)
        length_out.append(len(tmp_out_sen))
        tmp_out_sen.extend([0]*(de_len - len(tmp_out_sen)))
        idx_out_sen.append(tmp_out_sen)
    
    idx = np.random.choice(len(idx_in_sen))
    print("Orig data  :" , x[0][remain_idx[idx]])
    print("Index data :" , idx_in_sen[idx])
    
    print("Output Orig data  :" , x[1][remain_idx[idx]])
    print("Output Index data :" , idx_out_sen[idx])
    
    
    return np.array(idx_in_sen) , np.array(idx_out_sen) , np.stack(mask_in,axis=0) , np.stack(mask_out,axis=0) \
                , np.array(length_in) , np.array(length_out) , idx2word , word2idx , remain_idx

    
    
def transform_word(x , min_count = 3,max_len = 20):
    print("Min Count :" , min_count)
    print("Max Length :" , max_len)
    word_freq = defaultdict(lambda: 0)
    for xx in x:
        for sr in xx:
            # Record frequency of word
            for s in sr:
                word_freq[s] += 1
    
    idx2word = ["NULL" , "OOV" , "SOS" , "EOS"]
    idx2word.extend([ k for k,v in word_freq.items() if(v >= min_count and (k not in ["SOS" , "EOS"]) )])
    word2idx = dict([(k,i) for i,k in enumerate(idx2word)])
    print("Word Count :" , len(idx2word))
    
    idx_in_sen = []
    mask_in = []
    idx_out_sen = []
    mask_out = []
    remain_idx = []
    
    
    
    for c , (s1 , s2) in enumerate(zip(*x)):
        ## remove SOS , EOS
        s1 = s1[1:-1]
        
        if(len(s1) > max_len or len(s2) > max_len ):
            continue
        remain_idx.append(c)
        
        mask_in.append(np.zeros([max_len]))
        mask_in[-1][-len(s1)::] += 1
        tmp_in_sen = [0]*(max_len - len(s1))
        for s in s1:
            try:
                tmp_in_sen.append(word2idx[s])
            except:
                tmp_in_sen.append(1)
        idx_in_sen.append(tmp_in_sen)
        
        tmp_out_sen = []
        mask_out.append(np.zeros([max_len]))
        mask_out[-1][0:len(s2)] += 1
        for s in s2:
            try:
                tmp_out_sen.append(word2idx[s])
            except:
                tmp_out_sen.append(1)
        tmp_out_sen.extend([0]*(max_len - len(s2)))
        idx_out_sen.append(tmp_out_sen)
    
    return np.array(idx_in_sen) , np.array(idx_out_sen) , np.stack(mask_in,axis=0) , np.stack(mask_out,axis=0) \
                , idx2word , word2idx , remain_idx


def transform_gramma(x , idx , max_len = 20):
    
    gramma2idx = {}
    c = 1
    for gm in x:
        for g in gm:
            try:
                gramma2idx[g]
            except:
                gramma2idx[g] = c
                c+=1
    idx2gramma = ["NULL"]            
    idx2gramma.extend([k for k in gramma2idx.keys()])
    gramma2idx = dict([(k,i) for i,k in enumerate(idx2gramma)])
    print("Gramma Count :" , len(idx2gramma))
    
    idx_gramma = []
    for i in idx:
        tmp_arr = [gramma2idx[k] for k in x[i]]
        tmp_arr.extend([0]*(max_len-len(x[i])))
        idx_gramma.append(tmp_arr)
    
    return np.array(idx_gramma) , idx2gramma , gramma2idx


def transform_rhy(x , idx):
    c = 0
    rhy2idx = {}
    
    idx_rhy = []
    for i in idx:
        try:
            idx_rhy.append(rhy2idx[x[i]])
        except:
            rhy2idx[x[i]] = c
            c += 1
            idx_rhy.append(rhy2idx[x[i]])
    idx2rhy = list(rhy2idx.keys())
    idx2rhy.sort(key=lambda x : rhy2idx[x])
    print("Rhyme Count :" , len(idx2rhy))
    
    return np.array(idx_rhy) , rhy2idx , idx2rhy


class data_manager:
    
    def __init__(self,path,train=True):
        self.train_flag = train
        self.orig_data = []
        self.in_sen = []
        self.gramma = []
        self.length = []
        self.rhyme = []
        if train:
            self.out_sen = []
        if(type(path) == list):
            for p in path:
                self.__get_data(p , train_flag=train)
        else:
            self.__get_data(path , train_flag=train)
            
    def __get_data(self , path , train_flag):
        with open(path , "r") as f:
            x = f.readline()
            while(x != ""):
                x = x.split(",")
                
                self.orig_data.append(x[0].split())
                
                if(train_flag):
                    self.out_sen.append(x[1].split())
                ## Next line
                x = f.readline()

        print("Data count :" , len(self.orig_data))

        idx = np.random.choice(len(self.orig_data))
        print("\n### Data view ###")
        print("Original data  :" , self.orig_data[idx])
        if(train_flag):
            print("Output Sentence :" , self.out_sen[idx])
        

#     def __get_data(self , path , train_flag):
#         with open(path , "r") as f:
#             x = f.readline()
#             while(x != ""):
#                 x = x.replace("," , " ")
#                 x = x.split()
                
#                 s = 0
#                 c = 0
#                 while(x[c] != "EOS"):
#                     c += 1
#                 c += 1
#                 self.in_sen.append(x[s:c])
#                 s = c

#                 while(x[c] != "NOP"):
#                     c+=1
#                 self.gramma.append(x[s:c]) # NOP is not needed

#                 c += 1
#                 s = c
#                 self.rhyme.append(x[c])

#                 c += 2 # Skip NOE
#                 s = c
#                 self.length.append(int(x[c]))
                
#                 self.orig_data.append([_w for _w in x[0:c+2]])
                
#                 if(train_flag):
#                     c += 2 # Skip NOR
#                     s = c
#                     self.out_sen.append(x[s::])
#                 ## Next line
#                 x = f.readline()

#         print("Data count :" , len(self.in_sen))

#         idx = np.random.choice(len(self.in_sen))
#         print("\n### Data view ###")
#         print("Original data  :" , self.orig_data[idx])
#         print("Input sentence :" , self.in_sen[idx])
#         print("Gramma         :" , self.gramma[idx])
#         print("Length         :" , self.length[idx])
#         print("Rhyme          :" , self.rhyme[idx])
#         if(train_flag):
#             print("Output Sentence :" , self.out_sen[idx])

