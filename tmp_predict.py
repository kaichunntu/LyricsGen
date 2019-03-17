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
exp_folder = 

Encoder_max_len = 
Decoder_max_len = 
min_count = 
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
print("""
##################################################################################################
#######################################  Building Model  #########################################
##################################################################################################
""")

"""Building here"""
    
print("\nFinish Building!!\n")

print("""
##################################################################################################
######################################  Finish Building  #########################################
##################################################################################################
""")
# ## Training
print("""
##################################################################################################
#######################################  Start Infering  #########################################
##################################################################################################
""")

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



test_infer = infer_test_data(sess,infer_pred_idx_seq,test_agent.orig_data,word2idx,batch_size=1000)




print("Infer samples :")
for i in np.random.choice(len(test_agent.orig_data) , 10 , replace=False):
    print("  Input : " , " ".join(test_agent.orig_data[i]))
    print("  Infer : " , test_infer[i])
    print()




def save_infer_seg(data,name):
    path = os.path.join(exp_folder,name)
    print("Save at '{}'".format( path))
    with open( path, "w") as f:
        for s in data:
            if(len(s) == 0):
                s = np.random.choice(idx2word[4::])
            f.write(s+"\n")


save_infer_seg(test_infer,"{}_infer_seg.txt".format(output_file_name))