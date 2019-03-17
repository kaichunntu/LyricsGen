
# coding: utf-8

# In[1]:


import os
import time
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.contrib.framework import nest 
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import ops

sess_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True)
                         ,device_count={'GPU': 1})

from utils import exist_or_mkdir , data_manager , transform_orig

exp_folder = "DNN_ver0"
model_path = "model_para"
tmp_path = "tmp"
log_path = "log"


# In[3]:


exp_folder = exist_or_mkdir("./",exp_folder)
model_path = exist_or_mkdir(exp_folder,model_path)
tmp_path = exist_or_mkdir(exp_folder,tmp_path)
log_path = exist_or_mkdir(exp_folder,log_path)


# In[4]:


Encoder_max_len = 50
Decoder_max_len = 25
min_count = 3
print_interval = 1500
train_batch_size = 128
n_epoch = 50


# In[5]:


print("""
##################################################################################################
########################################  Loading Data  ##########################################
##################################################################################################
""")


# In[6]:


train_path = ["data/{}/train.csv".format(x) for x in ["all"]]
test_path = ["data/{}/test.csv".format(x) for x in ["all"]]

print("### Loading Train Data ###")
data_agent = data_manager(train_path , train=True)


print("### Loading Test Data ###")
test_agent = data_manager(test_path , train=False)


# In[7]:


print("\n### Preprocessing ###")
idx_in_sen , idx_out_sen , mask_in , mask_out , length_in , length_out , idx2word , word2idx , remain_idx =    transform_orig([data_agent.orig_data,data_agent.out_sen],min_count=min_count, max_len = [Encoder_max_len,Decoder_max_len])


# In[8]:


pickle.dump({"orig_word":[idx2word,word2idx] },
            open(os.path.join(tmp_path,"tokenizer.pkl") , "wb"))


# In[9]:


print("""
##################################################################################################
#######################################  Building Model  #########################################
##################################################################################################
""")
start_time = time.time()


# In[10]:


def attn_encode(_inputs,dim,length,b,reuse_flag=False,name=None):
    inputs = tf.reshape(_inputs,[-1,int(_inputs.shape[-1])])
    fc1 = tf.layers.dense(inputs , dim , activation=tf.nn.leaky_relu,name="{}_fc1".format(name),reuse=reuse_flag)
    fc2 = tf.layers.dense(fc1 , dim , activation=None,name="{}_fc2".format(name),reuse=reuse_flag)
    f = tf.layers.dense(fc2,dim,name="{}_fc3".format(name),reuse=reuse_flag)
    
    c = tf.matmul(f,fc2,transpose_b=True)
    mask = tf.concat([tf.ones([length,length]) , tf.zeros([length,(b-1)*length])],axis=-1)
    time = tf.constant(1,tf.int32)
    def body(time,mask):
        tmp = tf.concat([tf.zeros([length,length*time]),
                         tf.ones([length,length]),
                         tf.zeros([length,(b-time-1)*length])],
                        axis=-1)
        return time+1 , tf.concat([mask,tmp],axis=0)
    _ , mask = tf.while_loop(lambda t,*_:t<b,body,loop_vars=[time,mask])
    
    exp_c = tf.exp(c)*mask
    c = exp_c/(tf.reduce_sum(exp_c,axis=-1,keepdims=True)+1e-10)
    ws = c@fc2
    return tf.reshape(ws+fc1,[-1,length,dim])


# In[11]:


def deconv(inputs,encoder_vec,max_len,residual=None,dim=256,kernel_size=3,reuse_flag=True):
    print("Deconv inputs : " , inputs)
    with tf.name_scope("Deconv"):
        BA,MA,DIM = tf.unstack(tf.shape(inputs))
        vec_o = attn_encode(inputs,dim,MA,BA,reuse_flag=True,name="EN_layer0")
        f_vec = tf.layers.dense(tf.reshape(vec_o[:,-1,:],[BA,1,dim]),dim,activation=tf.nn.leaky_relu)
        f_vec = tf.concat([vec_o,tf.tile(f_vec,[1,MA,1])],axis=-1)
        z = tf.layers.dense(f_vec,128,activation=tf.nn.leaky_relu)
        z = tf.layers.dense(z,1,activation=None)
        z = tf.nn.softmax(z,axis=1)
        vec_o = tf.reduce_sum(vec_o*z,axis=1,keepdims=True)
        
#         vec_o = tf.layers.dense(inputs,int(encoder_vec.shape[-1]),activation=tf.nn.leaky_relu)
        print(vec_o)
        vec = tf.concat([tf.tile(vec_o,[1,max_len,1]),encoder_vec],axis=-1)
        attn_score = tf.layers.dense(vec,dim//2,activation=tf.nn.leaky_relu)
        attn_score = tf.nn.softmax(tf.layers.dense(attn_score,1,activation=None),axis=1)
        out = tf.reduce_sum(encoder_vec*attn_score,axis=1,keepdims=True)
    print("Deconv out :" , out)
    print("Deconv score :" , attn_score)
    return out , attn_score


# In[12]:


def wordclf(inputs,dim,embd_T,reuse_flag=False,training=True):
    with tf.name_scope("WordClf"):
        print("Wordclf inputs : " ,inputs)
        fc1 = tf.layers.dense(inputs,dim,activation=None,name="fc1",reuse=reuse_flag)
    #     fc1 = tf.layers.batch_normalization(fc1,trainable=training,training=training,
    #                                         reuse=reuse_flag,name="BN1")
        fc1 = tf.nn.leaky_relu(fc1)
        fc2 = tf.layers.dense(fc1,embd_dim,activation=None,reuse=reuse_flag,name="fc2")
        print(fc2)
        flat = tf.layers.flatten(fc2)
        print(flat)
        out = flat @ embd_T
    return out


# In[13]:


def dynamic_deconv(encoder,embd,embd_T,max_time):
    
    batch_size,max_len,encoder_dim = tf.unstack(tf.shape(encoder))
    
    emit_ta = nest.pack_sequence_as(int(embd.shape[0]),
                                    [tensor_array_ops.TensorArray(tf.float32,
                                                                  clear_after_read=False,
                                                                  size=0,
                                                                  dynamic_size=True,
                                                                  element_shape=tensor_shape.\
                                                                      TensorShape([None,len(idx2word)]))])
    emit_input = nest.pack_sequence_as(int(1),
                                           [tensor_array_ops.TensorArray(tf.int32,
                                                                         clear_after_read=False,
                                                                         size=0,
                                                                         dynamic_size=True,
                                                                         element_shape=tensor_shape.\
                                                                             TensorShape([None]))])
    
    emit_score = tensor_array_ops.TensorArray(tf.float32,
                                              clear_after_read=False,
                                              size=0,
                                              dynamic_size=True,
                                              element_shape=tensor_shape.\
                                                  TensorShape([None,None,1]))
    
    time = tf.constant(0,dtype=tf.int32)
    output_time = tf.constant(0,dtype=tf.int32)
    def initialize(batch_size,time,emit_input):
        for w in ["SOS"]:
            idx = tf.reshape(tf.constant(word2idx[w],dtype=tf.int32),[-1])
            idx = tf.tile(idx,[batch_size])
            emit_input = nest.map_structure(lambda ta,em: ta.write(time,em),emit_input,idx)
            time += 1
        return emit_input,time
    
    emit_input, time = initialize(batch_size,time,emit_input)
    
    def body(output_time,time,emit_input,emit_ta,emit_score):
        
#         inputs_idx = tf.transpose(emit_input.gather([time-1]),[1,0])
        inputs_idx = tf.transpose(emit_input.stack(),[1,0])
        print("input idx",inputs_idx)
        
        inputs_vec = tf.nn.embedding_lookup(embd,inputs_idx)
        output_vec , attn_s= deconv(inputs_vec,encoder,max_len,dim=256,reuse_flag=False)
        output_logits = wordclf(output_vec,300,embd_T,reuse_flag=False)
        
        next_idx = tf.argmax(output_logits,axis=-1,output_type=tf.int32)
        emit_input = nest.map_structure(lambda ta,em:ta.write(time,em),emit_input,next_idx)
        time += 1
        
        emit_ta = emit_ta.write(output_time,output_logits)
        emit_score = emit_score.write(output_time,attn_s)
        output_time += 1
        
        return output_time,time,emit_input,emit_ta,emit_score
    
    def condition(t,*_):
        return t<max_time
    
    _,_,emit_input,emit_ta,emit_score =             tf.while_loop(condition,body,
                          loop_vars=[output_time,time,emit_input,emit_ta,emit_score],
                          swap_memory=False)
        
    emit_input = tf.transpose(emit_input.stack(),[1,0])[:,1::]
    emit_ta = tf.transpose(emit_ta.stack(),[1,0,2])
    emit_score = tf.transpose(emit_score.stack(),[1,0,2,3])
    emit_score = tf.reshape(emit_score , [batch_size,-1,max_len])
    return emit_input,emit_ta,emit_score


# In[14]:


def get_grad_norm(_t_position,_p_position,pos_tensor):
    grad = tf.gradients(logits[0,_t_position,_p_position],pos_tensor)[0]
    return tf.sqrt(tf.reduce_sum(grad*grad,axis=-1))


# In[15]:


embd_dim = 200
L0_dim = 256
L1_dim = 256
L2_dim = 300
clf_dim = 256

g_conv = tf.Graph()
with g_conv.as_default() as g:
    with tf.name_scope("Input"):
        _xs = tf.placeholder(tf.int32,[None,None])
        _xs_length = tf.placeholder(tf.int32,[None])
        xs_mask = tf.sequence_mask(_xs_length,dtype=tf.float32)
        
        _ys = tf.placeholder(tf.int32,[None,None])
        ys_one_hot = tf.one_hot(_ys,depth=len(idx2word))
        _ys_length = tf.placeholder(tf.int32,[None])
        ys_mask = tf.sequence_mask(_ys_length,dtype=tf.float32)
        
        t_place = tf.placeholder(tf.int32,[])
        _exp_idx = tf.placeholder(tf.int32,[None])
        
        
        
    with tf.name_scope("Embedding"):
        _embd = tf.get_variable("Embedding",[len(idx2word),embd_dim],dtype=tf.float32,trainable=True,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        _embd_T = tf.transpose(_embd,[1,0])
        x_vector = tf.nn.embedding_lookup(_embd, _xs, max_norm=5)
        BatchSize , MAX_LEN , DIM = tf.unstack(tf.shape(x_vector))
        _exp_vec = tf.nn.embedding_lookup(_embd,_exp_idx)
        
    with tf.name_scope("DNN_encode"):
        attn_dnn0 = attn_encode(x_vector,L0_dim,MAX_LEN,BatchSize,name="EN_layer0")
#         attn_dnn1 = attn_encode(attn_dnn0,L1_dim,MAX_LEN,BatchSize,name="EN_layer1")
        attn_dnn1 = attn_dnn0
        
    with tf.name_scope("DNN_decode"):
        _pred , logits , _attn_map = dynamic_deconv(attn_dnn1,_embd,_embd_T,t_place)
    
    with tf.name_scope("Loss"):
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys_one_hot,logits=logits)*ys_mask
        ce = tf.reduce_sum(ce,axis=1,keepdims=False)/tf.cast(_ys_length,tf.float32)
        _loss = tf.reduce_mean(ce)
        
    with tf.name_scope("Train"):
        g_step = tf.Variable(0,dtype=tf.int32,trainable=False,name="Global_step")
        lr = tf.train.exponential_decay(5e-4,g_step,2000,0.95,staircase=True)
        opt = tf.train.AdamOptimizer(lr)
        allgrads = opt.compute_gradients(_loss) 
        clip_grads = [ ( tf.clip_by_average_norm(grad,10) ,var) for grad , var in allgrads]
        _update = opt.apply_gradients(clip_grads,global_step=g_step)
#         _update = opt.minimize(_loss,global_step=g_step)
        
    with tf.name_scope("Gradient"):
        _t_position = tf.placeholder(tf.int32)
        _p_position = _pred[0,_t_position]
        xs_vector_gradnorm = get_grad_norm(_t_position,_p_position,x_vector)
        dnn1_gradnorm = get_grad_norm(_t_position,_p_position,attn_dnn0)
        dnn2_gradnorm = get_grad_norm(_t_position,_p_position,attn_dnn1)
    
    all_var = tf.trainable_variables()
    _init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=8,var_list=tf.trainable_variables())
#     tf.summary.FileWriter(log_path,graph=g)


# In[16]:


print("Consume time : {:.2f}".format(time.time()-start_time))
print()
def calculate_var(var):
    d = 0
    for w in var:
        tmp_d = 1
        for ww in w.shape:
            tmp_d *= int(ww)
        d += tmp_d
    print("Var Count :" , d)
    return d

calculate_var(all_var)


# In[17]:


print("""
##################################################################################################
#######################################  Start Infering  #########################################
##################################################################################################\n
""")


# In[18]:


sess = tf.Session(graph=g_conv,config=sess_opt)
sess.run(_init)


# In[19]:


def get_batch(i):
    tmp_end = max(length_in[i])
    out_max_len = max(length_out[i])
    my_dict = {
        _xs:idx_in_sen[i,:tmp_end],
        _ys:idx_out_sen[i,:out_max_len],
        _ys_length:length_out[i],
        t_place:out_max_len
    }
    return my_dict


# In[20]:


def evaluate_batch(sess,_pred,count=3):
    idx = np.random.choice(idx_in_sen.shape[0],[count])
    tmp_max_len = max(length_in[idx])
    my_dict = {
        _xs:idx_in_sen[idx,:tmp_max_len],
        t_place:30
    }
    pred = sess.run(_pred , feed_dict=my_dict)
    
    word_seq = []
    for i in range(3):
        idx_sen = pred[i]
        tmp = []
        for t in range(Decoder_max_len-1):
            if(idx_sen[t] == 3):
                break
            tmp.append(idx2word[idx_sen[t]])
        word_seq.append(tmp)
    
    print("Max length :" , tmp_max_len)
    for i in range(3):
        print("  Input word  :" , data_agent.orig_data[remain_idx[idx[i]]])
        print("  Input index :" , idx_in_sen[idx[i],:tmp_max_len])
        print("  Ground word :" , data_agent.out_sen[remain_idx[idx[i]]])
        print("    Output    :" , word_seq[i])
        print()


# In[20]:


batch_size = train_batch_size
n_step = idx_in_sen.shape[0]//batch_size

r_index = np.arange(idx_in_sen.shape[0])
loss_list = []
try:
    for e in range(1,40+1):
        np.random.shuffle(r_index)
        start_time = time.time()
        start = 0
        for s in range(n_step):
            idx = r_index[start:start+batch_size]
            _,l = sess.run([_update,_loss] , feed_dict=get_batch(idx))
            start += batch_size
            print("step {:>5d} loss : {:>9.4f} time : {:>7.2f}".format(s,l,time.time()-start_time) , end="\r")
            loss_list.append(l)
            if s % print_interval == 0:
                print("step {:>5d} loss : {:>9.4f} time : {:>7.2f}".format(s,l,time.time()-start_time) , end="\n")
                evaluate_batch(sess,_pred,3)
        print("\nEpoch {0:>3d}/{1:d} loss : {2:>9.4f} time : {3:>8.2f}".format(e,n_epoch,l,time.time()-start_time))

        evaluate_batch(sess,_pred,3)

        if e%4 == 0:
            saver.save(sess,os.path.join(model_path,"model_{}.ckpt".format(e)))
except KeyboardInterrupt :
    print("Interrupt .. Save lastest model...")
    saver.save(sess,os.path.join(model_path,"model_{}.ckpt".format("lastest")))
    print()
    
print("Save loss history...")
pickle.dump(loss_list,open(os.path.join(log_path,"loss.pkl") , "wb"))


# In[ ]:


print("\n### Training Finished!! ###\n")
