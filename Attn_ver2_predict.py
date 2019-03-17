import os
import sys
import time
import pickle

from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
sess_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True)
                         ,device_count={'GPU': 1})

from matplotlib import pyplot as plt

from utils import exist_or_mkdir , data_manager , transform_orig


########## Setting ##########
exp_folder = "Attn_ver2"

Encoder_max_len = 60
Decoder_max_len = 30
min_count = 3
#############################

model_path = "model_para"
tmp_path = "tmp"
log_path = "log"


# In[2]:


exp_folder = exist_or_mkdir("./",exp_folder)
model_path = exist_or_mkdir(exp_folder,model_path)
tmp_path = exist_or_mkdir(exp_folder,tmp_path)
log_path = exist_or_mkdir(exp_folder,log_path)


# ## Loading data

# In[3]:

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]
try:
    model_index = sys.argv[3]
except:
    model_index = None



print("### Loading Test Data ###")
test_agent = data_manager(input_file_name , train=False)


# ## Preprocessing and Padding

# pickle.dump({"orig_word":[idx2word,word2idx] },
#             open(os.path.join(tmp_path,"tokenizer.pkl") , "wb"))

idx2word,word2idx = pickle.load(open(os.path.join(tmp_path,"tokenizer.pkl") , "rb"))["orig_word"]

##################################################################################################
#######################################  Building Model  #########################################
##################################################################################################

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


# In[12]:


def attend_vector(inputs , state , mask , name):
    with tf.name_scope("Attention"):
        state = tf.tile(tf.expand_dims(state , axis=1) , [1,tf.shape(inputs)[1],1])
        concat_vec = tf.concat([inputs,state],axis=-1)
        fc1 = ly.fully_connected(concat_vec,256,activation_fn=tf.nn.leaky_relu,biases_initializer=None,
                                 scope="Attn_{}_1".format(name),reuse=tf.AUTO_REUSE)
        fc2 = ly.fully_connected(fc1,128,activation_fn=tf.nn.leaky_relu,biases_initializer=None,
                                 scope="Attn_{}_2".format(name),reuse=tf.AUTO_REUSE)
        fc3 = ly.fully_connected(fc1,1,activation_fn=None,biases_initializer=None,
                                 scope="Attn_{}_3".format(name),reuse=tf.AUTO_REUSE)
        score = tf.nn.softmax(fc3 , axis=1)
        ## define my softmax
#         exp_fc3 = tf.exp(fc3)*mask
#         exp_sum = tf.reduce_sum(exp_fc3,axis=1,keepdims=True)
#         score = exp_fc3/exp_sum
    
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


def word_clf(inputs,dim,embd):
    fc1 = ly.fully_connected(inputs,dim,activation_fn=tf.nn.leaky_relu,scope="clf_fc1",reuse=tf.AUTO_REUSE)
    fc2 = ly.fully_connected(fc1,int(embd.shape[0]),activation_fn=None,scope="clf_fc2",reuse=tf.AUTO_REUSE)
    return fc2@embd


# In[14]:


def mask_catece(x):
    logit = x[0]
    idx = x[1]
    ce = []
    for t in range(Decoder_max_len-1):
        ce.append( tf.log(tf.nn.embedding_lookup(logit[t],idx[t])+1e-10) )
    return tf.stack(ce)


# In[21]:


Seq_g = tf.Graph()
embd_dim = 256
L0_dim = 384
L1_dim = 512
L2_dim = 512
clf_dim = 300
with Seq_g.as_default():
    with tf.name_scope("Input"):
        _in = tf.placeholder(tf.int32,[None,None])
        _in_mask = tf.placeholder(tf.float32,[None,None])
        in_mask = tf.expand_dims(_in_mask,axis=-1)
        
        _in_length = tf.placeholder(tf.int32,[None])
        
        _out = tf.placeholder(tf.int32,[None,Decoder_max_len])
        _out_mask = tf.placeholder(tf.float32,[None,Decoder_max_len])
        gt = _out[:,1::]
        gt_mask = _out_mask[:,1::]
        
        schedual_rate = tf.random_uniform([Decoder_max_len],maxval=1.0)
        schedual_th = tf.placeholder(tf.float32)
        infer_start = tf.ones([tf.shape(_in)[0]],dtype=tf.int32)
        
    with tf.name_scope("Embedding"):
        ## word embedding
        _embd = tf.Variable(tf.truncated_normal([len(idx2word) , embd_dim],stddev=0.1),name="Word_Embd")
        _embd_T = tf.transpose(_embd,[1,0])
        x_vector = tf.nn.embedding_lookup(_embd,_in,max_norm=5)
        y_vector = tf.nn.embedding_lookup(_embd,_out,max_norm=5)
        
    
    
    with tf.name_scope("Encoder"):
        e_cell0 = tf.contrib.rnn.LSTMCell(384,name="E_layer_0",reuse=False)
        e_cell1 = tf.contrib.rnn.LSTMCell(512,name="E_layer_1",reuse=False)
        
        E_layer_0 , E_state_0= tf.nn.dynamic_rnn(e_cell0,x_vector,sequence_length=_in_length,dtype=tf.float32)
        E_layer_1 , E_state_1= tf.nn.dynamic_rnn(e_cell1,E_layer_0,sequence_length=_in_length,dtype=tf.float32)
        
    with tf.name_scope("Decoder"):
        
        D_layer_0 , D_state_0 = Encoder(y_vector,384,"rnn/E_layer_0",init_state=E_state_0,reuse=True,
                                        t_len=Decoder_max_len-1,stack_flag=True)
        D_layer_1 , D_state_1 = Encoder(D_layer_0,512,"rnn/E_layer_1",init_state=E_state_1,reuse=True,
                                        t_len=Decoder_max_len-1,stack_flag=False)
        
        D_layer_1 , D_state_1 , D_score = attn_Decoder(D_layer_1,E_layer_1,in_mask,512,name="Attn_D_layer_1",
                                                       init_state=E_state_1,t_len=Decoder_max_len-1,stack_flag=False)
        
        output_seq = []
        for t in range(Decoder_max_len-1):
            choice_input = D_layer_1[t]
            out = word_clf(choice_input,300,_embd_T)
            output_seq.append(out)
        _logits = tf.stack(output_seq,axis=1)
        _prob = tf.nn.softmax(_logits,axis=-1)
        
        
    with tf.name_scope("Loss"):
#         cross_entropy_0 = tf.map_fn(mask_catece,elems=(_prob,gt),dtype=tf.float32)
#         cross_entropy = tf.reduce_sum(cross_entropy_0*gt_mask,axis=-1)/tf.reduce_sum(gt_mask,axis=-1)
#         _loss = -tf.reduce_mean(cross_entropy)

        gt = tf.one_hot(gt,depth=len(idx2word),dtype=tf.float32)
        cross_entropy_0 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(gt,[-1,len(idx2word)]),
                                                                     logits=tf.reshape(_logits,[-1,len(idx2word)]))
        cross_entropy_1 = tf.reshape(cross_entropy_0,[-1,Decoder_max_len-1])
        cross_entropy = tf.reduce_sum(cross_entropy_1*gt_mask,axis=-1)/tf.reduce_sum(gt_mask,axis=-1)
        _loss = tf.reduce_mean(cross_entropy)
        
    with tf.name_scope("Train_strategy"):
        opt = tf.train.AdamOptimizer(1e-4)
        _update = opt.minimize(_loss)
    
    with tf.name_scope("Inference"):
        ## start at Encoder layer 2 : E_layer2
        infer_out = tf.nn.embedding_lookup(_embd,infer_start)
        infer_state_0 = E_state_0
        infer_state_1 = E_state_1
        infer_state_2 = E_state_1
        
        infer_score_seq = []
        infer_pred_idx_seq = []
        infer_logits_seq = []
        for t in range(Decoder_max_len-1):
            tmp = Encoder([infer_out],384,"rnn/E_layer_0",init_state=infer_state_0,reuse=True,
                          t_len=1,stack_flag=False)
            infer_layer_0 , infer_state_0 = tmp
            
            
            tmp = Encoder(infer_layer_0,512,"rnn/E_layer_1",init_state=infer_state_1,reuse=True,
                          t_len=1,stack_flag=False)
            infer_layer_1 , infer_state_1 = tmp
            
            tmp = attn_Decoder(infer_layer_1,E_layer_1,in_mask,512,name="Attn_D_layer_1",
                               init_state=infer_state_2,t_len=1,reuse=True,stack_flag=False)
            
            infer_layer_2 , infer_state_2 , infer_score = tmp
            
            infer_score_seq.append(infer_score)
            
            infer_out = word_clf(infer_layer_1[0],300,_embd_T)
            infer_logits_seq.append(infer_out)
            
            out_index = tf.argmax(infer_out,axis=1)
            infer_pred_idx_seq.append(out_index)
            infer_out = tf.nn.embedding_lookup(_embd , out_index)
            
        infer_pred_idx_seq = tf.stack(infer_pred_idx_seq,axis=1)
        infer_logits = tf.stack(infer_logits_seq,axis=1)
        infer_prob = tf.nn.softmax(infer_logits,axis=-1)
    
    tf.summary.FileWriter(log_path,graph=Seq_g)
    _init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10,var_list=tf.global_variables())
    
print("\nFinish Building!!\n")


##################################################################################################
######################################  Finish Building  #########################################
##################################################################################################

##################################################################################################
#######################################  Start Infering  #########################################
##################################################################################################


print("### Start Infer ###\n")
if(model_index):
    path = os.path.join(model_path,"model_{}.ckpt".format(model_index))
else:
    path = tf.train.latest_checkpoint(model_path)
sess = tf.Session(graph=Seq_g,config=sess_opt)

saver.restore(sess,path)

# ## Inference

# In[39]:


def infer_test_data(sess,_pred,x,word2idx,batch_size=1000):
    start = 0
    pred_word_seq = []
    while(start < len(x)):
        print("Count :" , start , end="\r")
        batch_idx_sen = []
        batch_length = []
        batch_mask = []
        max_len = 0
        for s in x[start : start+batch_size]:
            l = len(s)-1
            if(l>max_len):
                max_len = l
        
        for s in x[start : start+batch_size]:
            arr = []
            ## s[1::] : remove first word "SOS"
            batch_mask.append( np.zeros([max_len]))
            batch_mask[-1][0:len(s[1::])] += 1 
            batch_length.append(len(s[1::]))
            for ss in s[1::]:
                try:
                    arr.append(word2idx[ss])
                except:
                    arr.append(1)
            arr.extend([0]*(max_len-len(s[1::])))
            batch_idx_sen.append(arr)
        batch_idx_sen = np.array(batch_idx_sen)
        batch_length = np.array(batch_length)
        batch_mask = np.stack(batch_mask)
        
        pred_sen = sess.run(_pred,feed_dict={
            _in:batch_idx_sen,
            _in_length:batch_length,
            _in_mask:batch_mask
        })
        
        for i in range(batch_size):
            idx_sen = pred_sen[i]
            tmp = []
            for t in range(Decoder_max_len-1):
                if(idx_sen[t] == 3):
                    break
                elif(idx_sen[t] == 1):
                    tmp.append(np.random.choice(idx2word))
                else:
                    tmp.append(idx2word[idx_sen[t]])
            pred_word_seq.append(" ".join(tmp))
        start += batch_size
        
    return pred_word_seq


# In[40]:


test_infer = infer_test_data(sess,infer_pred_idx_seq,test_agent.orig_data,word2idx,batch_size=1000)


# In[46]:


print("Infer samples :")
for i in np.random.choice(len(test_agent.orig_data) , 10 , replace=False):
    print("  Input : " , " ".join(test_agent.orig_data[i]))
    print("  Infer : " , test_infer[i])
    print()


# In[53]:


def save_infer(data,name):
    path = os.path.join(exp_folder,name)
    print("Save at '{}'".format( path))
    with open( path, "w") as f:
        for s in data:
            s = "".join(s.split())
            if(len(s) == 0):
                s = np.random.choice(idx2word[4::])
            f.write(s+"\n")

def save_infer_seg(data,name):
    path = os.path.join(exp_folder,name)
    print("Save at '{}'".format( path))
    with open( path, "w") as f:
        for s in data:
            if(len(s) == 0):
                s = np.random.choice(idx2word[4::])
            f.write(s+"\n")


# In[54]:


save_infer(test_infer,"{}_infer_output.txt".format(output_file_name))
save_infer_seg(test_infer,"{}_infer_seg.txt".format(output_file_name))