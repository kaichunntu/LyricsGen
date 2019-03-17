
# coding: utf-8

# In[1]:


import os
import time
import pickle

from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
sess_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True)
                         ,device_count={'GPU': 1})

from matplotlib import pyplot as plt

from utils import exist_or_mkdir , data_manager , transform_word , transform_gramma , transform_rhy

exp_folder = "ALL_ver0"
model_path = "model_para"
tmp_path = "tmp"
log_path = "log"



exp_folder = exist_or_mkdir("./",exp_folder)
model_path = exist_or_mkdir(exp_folder,model_path)
tmp_path = exist_or_mkdir(exp_folder,tmp_path)
log_path = exist_or_mkdir(exp_folder,log_path)


max_len = 20
min_count = 3

print("\n### Loading Train Data ###")
data_agent = data_manager("data/all/train.csv" , train=True)

print("\n### Loading Test Data ###")
test_agent = data_manager("data/all/test.csv" , train=False)


## Preprocessing and Padding
print("\n### Preprocessing and Padding ###")
start_t = time.time()
idx_in_sen , idx_out_sen , mask_in , mask_out , idx2word , word2idx , remain_idx =     transform_word([data_agent.in_sen , data_agent.out_sen],min_count=min_count,max_len = max_len)


idx_gramma , idx2gramma , gramma2idx = transform_gramma(data_agent.gramma , remain_idx,max_len = max_len)
length = np.array(data_agent.length)[remain_idx]
print("Max length : " , max(length))
idx_rhy , rhy2idx , idx2rhy = transform_rhy(data_agent.rhyme,remain_idx)
idx_rhy.shape , length.shape , idx_in_sen.shape , idx_out_sen.shape , idx_gramma.shape


pickle.dump({"word":[idx2word,word2idx] , "rhyme":[idx2rhy,rhy2idx] , "gramma":[idx2gramma,gramma2idx]},
            open(os.path.join(tmp_path,"tokenizer.pkl") , "wb"))

print("Finish !! time : {:>5.2f}".format(time.time()-start_t))


## Building model
print("\n### Build Model... ###")
start_t = time.time()

def Encoder(inputs , dim , name , init_state=None , t_len=20 , reuse=False , stack_flag=False):
    cell = tf.contrib.rnn.LSTMCell(dim,name=name,reuse=reuse)
    if init_state:
        state = init_state
    else:
        state = [tf.zeros([tf.shape(inputs)[0] , cell.state_size[0]]),
                 tf.zeros([tf.shape(inputs)[0] , cell.state_size[1]])]
    output_seq = []
    for t in range(t_len):
        if stack_flag:
            out , state = cell(inputs[:,t] , state)
        else:
            out , state = cell(inputs[t] , state)
        output_seq.append(out)
    
    return output_seq , state


# In[13]:


def attend_vector(inputs , state , mask , name):
    with tf.name_scope("Attention"):
        state = tf.tile(tf.expand_dims(state , axis=1) , [1,tf.shape(inputs)[1],1])
        concat_vec = tf.concat([inputs,state],axis=-1)
        fc1 = ly.fully_connected(concat_vec,128,activation_fn=tf.nn.leaky_relu,biases_initializer=None,
                                 scope="Attn_{}_1".format(name),reuse=tf.AUTO_REUSE)
        fc2 = ly.fully_connected(fc1,64,activation_fn=tf.nn.leaky_relu,biases_initializer=None,
                                 scope="Attn_{}_2".format(name),reuse=tf.AUTO_REUSE)
        fc3 = ly.fully_connected(fc1,1,activation_fn=None,biases_initializer=None,
                                 scope="Attn_{}_3".format(name),reuse=tf.AUTO_REUSE)
        score = tf.nn.softmax(fc3*mask , axis=1)
    
    return score , tf.reduce_sum(inputs*score , axis=1)

def attn_Encoder(inputs , mask , dim , name , init_state=None , t_len=20 , reuse=False):
    cell = tf.contrib.rnn.LSTMCell(dim,name=name,reuse=reuse)
    if init_state:
        state = init_state
    else:
        state = [tf.zeros([tf.shape(inputs)[0] , cell.state_size[0]]),
                 tf.zeros([tf.shape(inputs)[0] , cell.state_size[1]])]
    output_seq = []
    score_seq = []
    for t in range(t_len):
        score , attn_vec = attend_vector(inputs,state[1],mask,name="Encode")
        out , state = cell(attn_vec,state)
        output_seq.append(out)
        score_seq.append(score)
    
    return output_seq , state , score_seq 


def attn_Decoder(inputs , inputs_E , mask , dim , name , init_state=None , t_len=20 , reuse=False , stack_flag=False):
    cell = tf.contrib.rnn.LSTMCell(dim,name=name,reuse=reuse)
    if init_state:
        state = init_state
    else:
        state = [tf.zeros([tf.shape(inputs)[0] , cell.state_size[0]]),
                 tf.zeros([tf.shape(inputs)[0] , cell.state_size[1]])]
    output_seq = []
    score_seq = []
    for t in range(t_len):
        score , attn_vec = attend_vector(inputs_E,state[1],mask,name="Decode")
        if stack_flag:
            attn_vec = tf.concat([attn_vec,inputs[:,t]] , axis=-1)
        else:
            attn_vec = tf.concat([attn_vec,inputs[t]] , axis=-1)
        out , state = cell(attn_vec,state)
        output_seq.append(out)
        score_seq.append(score)
    
    return output_seq , state , score_seq 


# In[13]:


def word_clf(inputs,embd):
    fc1 = ly.fully_connected(inputs,128,activation_fn=tf.nn.leaky_relu,scope="clf_fc1",reuse=tf.AUTO_REUSE)
    fc2 = ly.fully_connected(fc1,128,activation_fn=None,scope="clf_fc2",reuse=tf.AUTO_REUSE)
    return fc2@embd


# In[14]:


def mask_catece(x):
    logit = x[0]
    idx = x[1]
    ce = []
    for t in range(max_len-1):
        ce.append( tf.log(tf.nn.embedding_lookup(logit[t],idx[t])+1e-10) )
    return tf.stack(ce)


# In[15]:


Seq_g = tf.Graph()
embd_dim = 128
embd_gra_dim = 32
embd_rhy_dim = 16
embd_len_dim = 16

with Seq_g.as_default():
    with tf.name_scope("Input"):
        _in = tf.placeholder(tf.int32,[None,max_len])
        _in_mask = tf.placeholder(tf.float32,[None,max_len])
        in_mask = tf.expand_dims(_in_mask,axis=-1)
        
        _out = tf.placeholder(tf.int32,[None,max_len])
        _out_mask = tf.placeholder(tf.float32,[None,max_len])
        
        gt = _out[:,1::]
        gt_mask = _out_mask[:,1::]
        
        _out_gra = tf.placeholder(tf.int32,[None,max_len])
        _out_len = tf.placeholder(tf.float32,[None])
        _out_rhy = tf.placeholder(tf.int32,[None])
        
        infer_start = tf.placeholder(tf.int32,[None])
        
    with tf.name_scope("Embedding"):
        ## word embedding
        _embd = tf.Variable(tf.truncated_normal([len(idx2word) , embd_dim],stddev=0.1),name="Word_Embd")
        _embd_T = tf.transpose(_embd,[1,0])
        x_vector = tf.nn.embedding_lookup(_embd,_in,max_norm=5)
        y_vector = tf.nn.embedding_lookup(_embd,_out,max_norm=5)
        
        ## gramma embedding
        _embd_gra = tf.Variable(tf.truncated_normal([len(idx2gramma) , embd_gra_dim],stddev=0.1),name="Gramma_Embd")
        gra_vector = tf.nn.embedding_lookup(_embd_gra,_out_gra,max_norm=1)
        
        ## rhyme embedding
        _embd_rhy = tf.Variable(tf.truncated_normal([len(idx2rhy) , embd_rhy_dim],stddev=0.1),name="Rhyme_Embd")
        rhy_vector = tf.nn.embedding_lookup(_embd_rhy,_out_rhy,max_norm=1)
        
        ## length embedding
        _embd_len = tf.Variable(tf.truncated_normal([1 , embd_len_dim],stddev=0.1),name="Rhyme_Embd")
        
        len_vector = tf.matmul(tf.expand_dims(_out_len,axis=1),_embd_len)
        time_vector = [len_vector]
        for _ in range(max_len-1):
            len_vector = ly.fully_connected(len_vector,embd_len_dim,activation_fn=None,
                                            scope="Time_transform",reuse=tf.AUTO_REUSE)
            time_vector.append(len_vector)
        time_vector = tf.stack(time_vector,axis=1)
        
        var_vector = tf.concat([gra_vector,time_vector,
                                tf.tile(tf.expand_dims(rhy_vector,axis=1),[1,max_len,1])] ,axis=-1)
        var_vector = ly.fully_connected(var_vector,64,activation_fn=tf.nn.leaky_relu,biases_initializer=None,
                                        scope="Var_transform" , reuse=tf.AUTO_REUSE)
        
        
    with tf.name_scope("Encoder"):
        E_layer1 , e_state_1 = Encoder(x_vector,256,t_len=max_len,name="E_layer_1",stack_flag=True)
        E_layer2 , e_state_2 , E_score_seq = attn_Encoder(tf.stack(E_layer1,axis=1),in_mask,256,
                                                          t_len=max_len,name="ATTN_E_layer2")
        
        
    with tf.name_scope("Decoder"):
        concat_out_vector = tf.concat([y_vector,var_vector],axis=-1)
        D_layer0 , d_state_0 = Encoder(concat_out_vector,128,name="D_layer0",t_len=max_len,stack_flag=True)
        D_layer1 , d_state_1 = Encoder(D_layer0,256,init_state=e_state_1,
                                       t_len=max_len,name="E_layer_1",reuse=True,stack_flag=False)
        D_layer2 , d_state_2 , D_score_seq = attn_Decoder(D_layer1,tf.stack(E_layer2,axis=1),in_mask,256,init_state=e_state_2,
                                                          name="ATTN_D_layer2",stack_flag=False)
        
        output_seq = []
        for t in range(max_len):
            output_seq.append(word_clf(D_layer2[t],_embd_T))
        _logits = tf.stack(output_seq,axis=1)
        _prob = tf.nn.softmax(_logits,axis=-1)
        
    with tf.name_scope("Loss"):
        cross_entropy_0 = tf.map_fn(mask_catece,elems=(_prob,gt),dtype=tf.float32)
        cross_entropy = tf.reduce_sum(cross_entropy_0*gt_mask,axis=-1)/tf.reduce_sum(gt_mask,axis=-1)
        _loss = -tf.reduce_mean(cross_entropy)
        
    with tf.name_scope("Train_strategy"):
        opt = tf.train.AdamOptimizer(1e-4)
        _update = opt.minimize(_loss)
    
    with tf.name_scope("Inference"):
        ## start at Encoder layer 2 : E_layer2
        infer_out = tf.nn.embedding_lookup(_embd,infer_start)
        infer_state_1 = e_state_1
        infer_state_2 = e_state_2
        
        infer_pred_idx_seq = []
        infer_logits_seq = []
        for t in range(max_len):
            infer_concat_vec = tf.concat([infer_out,var_vector[:,t]] , axis=-1)
            if t==0:
                tmp = Encoder([infer_concat_vec],128,name="D_layer0",t_len=1,reuse=True,stack_flag=False)
            else:
                tmp = Encoder([infer_concat_vec],128,init_state=infer_state_0,name="D_layer0",t_len=1,
                              reuse=True,stack_flag=False)
            infer_layer0 = tmp[0]
            infer_state_0 = tmp[1]
            
            tmp = Encoder(infer_layer0,256,init_state=infer_state_1,
                          t_len=1,name="E_layer_1",reuse=True,stack_flag=False)
            infer_layer1 = tmp[0]
            infer_state_2 = tmp[1]
            
            tmp = attn_Decoder(infer_layer1,tf.stack(E_layer2,axis=1),in_mask,256,init_state=infer_state_2,
                               t_len=1,name="ATTN_D_layer2",reuse=True,stack_flag=False)
            infer_layer2 , infer_state_2 , infer_score_seq = tmp
            
            infer_out = word_clf(infer_layer2[0],_embd_T)
            infer_logits_seq.append(infer_out)
            
            out_index = tf.argmax(infer_out,axis=1)
            infer_pred_idx_seq.append(out_index)
            infer_out = tf.nn.embedding_lookup(_embd , out_index)
            
        infer_pred_idx_seq = tf.stack(infer_pred_idx_seq,axis=1)
        infer_logits = tf.stack(infer_logits_seq,axis=1)
        infer_prob = tf.nn.softmax(infer_logits,axis=-1)
        
    _init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)
    
print("Finish Building!!  time : {:>5.2f}\n".format(time.time()-start_t))


# ## Training

# In[21]:


print("### Start training ###")

sess = tf.Session(graph=Seq_g,config=sess_opt)
sess.run(_init)


# In[18]:


def get_batch(i):
    my_dict = {
        _in:idx_in_sen[i],
        _in_mask:mask_in[i],
        _out:idx_out_sen[i],
        _out_mask:mask_out[i],
        _out_gra:idx_gramma[i],
        _out_len:length[i],
        _out_rhy:idx_rhy[i]
    }
    return my_dict


# In[26]:


batch_size = 450
n_epoch = 60
n_step = idx_in_sen.shape[0]//batch_size

r_index = np.arange(idx_in_sen.shape[0])
loss_list = []

for e in range(1,n_epoch+1):
    np.random.shuffle(r_index)
    start_time = time.time()
    start = 0
    for s in range(n_step):
        idx = r_index[start:start+batch_size]
        _,l = sess.run([_update,_loss] , feed_dict=get_batch(idx))
        start += batch_size
        print("step {:>5d} loss : {:>9.4f} time : {:>7.2f}".format(s,l,time.time()-start_time) , end="\r")
    
    loss_list.append(l)
    print("Epoch {0:>3d}/{1:d} loss : {2:>9.4f} time : {3:>8.2f}".format(e,n_epoch,l,time.time()-start_time))
    if e%3 == 0:
        saver.save(sess,os.path.join(model_path,"model_{}.ckpt".format(e)))
        pickle.dump(loss_list,open(os.path.join(log_path,"loss.pkl") , "wb"))

