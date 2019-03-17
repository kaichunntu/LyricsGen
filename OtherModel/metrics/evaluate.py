
import os
import sys
import numpy as np





def read_infer_feature(path):
    pos_arr = []
    rhy_arr = []
    count_arr = []
    with open(path,"r") as f:
        x = f.readline()
        while(x!=""):
            x = x.split(",")
            pos_arr.append(x[0].split(" "))
            rhy_arr.append(x[2].strip())
            count_arr.append(x[1])
            x = f.readline()
            
    return pos_arr , rhy_arr , count_arr
            
def read_gt_feature(path):
    pos_arr = []
    rhy_arr = []
    count_arr = []
    with open(path,"r") as f:
        x = f.readline()
        while(x!=""):
            x = x.split(" EOS ")
            if(x[1].find("NOP")>=0):
                x = x[1].strip()
                x = x.split(" NOP")
                pos_arr.append(x[0].split())
                
            if(x[1].find("NOE")>=0):
                x = x[1].strip()
                x = x.split(" NOE")
                rhy_arr.append(x[0])
                
            if(x[1].find("NOR")>=0):
                x = x[1].strip()
                x = x.split(" NOR")
                count_arr.append(x[0])
            x = f.readline()
    
    return pos_arr , rhy_arr , count_arr

def evaluate_score(ground,pred):
    assert len(ground) == len(pred)
    count = 0
    for g,p in zip(ground,pred):
        if(g == p):
            count += 1
    
    return count/len(ground)

def evaluate_pos_score(ground,pred):
    assert len(ground) == len(pred)
    count = 0
    all_count = 0
    for gg,pp in zip(ground,pred):
        lg = len(gg)
        lp = len(pp)
        max_len = max([lg,lp])
        min_len = min([lg,lp])
        for i in range(min_len):
            if(gg[i] == pp[i]):
                count += 1
        all_count += max_len
    
    return count/all_count

def jaccard_index(sample):
    score = 0
    for i,(a,b) in enumerate(sample):
        count = 0
        la = len(a)
        lb = len(b)
        for i in a:
            if(i in b):
                count += 1
            score += count/(la+lb-count)
    return score/len(sample)

            
if __name__ == "__main__":
    pred_data = read_infer_feature(sys.argv[1])
    gt_data = read_gt_feature( "./data/{}/test.csv".format( sys.argv[2] ) )
    
    print("\nPOS   baseline : 0.55")
    try:
        print("POS   score : {:5>.4f}".format( evaluate_pos_score(gt_data[0],pred_data[0])) )
    except:
        print("No POS test")
        
    print("\nRhyme baseline : 0.86")
    try:
        print("Rhyme score : {:5>.4f}".format( evaluate_score(gt_data[1],pred_data[1])) )
    except:
        print("No Rhyme test")
        
    print("\nCount baseline : 0.98")
    try:
        print("Count score : {:5>.4f}".format( evaluate_score(gt_data[2],pred_data[2])) )
    except:
        print("No Count test")
    
    
    try:
        infer_data = []
        with open(sys.argv[3],"r") as f:
            x = f.readline()
            while(x != ""):
                infer_data.append(x.split())
                x = f.readline()
        
        sample_arr = []
        total_count = len(infer_data)
        for _ in range(3000):
            tmp = []
            for i in np.random.choice(total_count,2,replace=False):
                tmp.append(infer_data[i])
            sample_arr.append(tmp)
        print("\nJaccard Index baselen : 0.25")
        print("Jaccard score : {:5>.4f}".format(jaccard_index(sample_arr)))
    except:
        print("No Jaccard test")
    print("\nEnd\n")