import argparse
import collections
import pickle
import math
import os

from nltk import bigrams, trigrams
import tensorflow as tf
import jieba

def dd():
    return collections.defaultdict(int)

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", " <eos> ").split()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_data', default='output.csv')
    args = parser.parse_args()
    
    f = open('trigram_model.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    
    f = open('w2i.pkl', 'rb')
    word_to_id = pickle.load(f)
    f.close()
    
    with open(args.testing_data, 'r') as f:
        if len(collections.Counter(f.read().replace('\n', '').replace(' ', '').strip())) <= 10:
            print ('LM score: 0')
            return

    with open(args.testing_data, 'r') as f:
        cnt_s = 0
        prob_all = 0
        for line in f:
            prob_s = 0
            cnt = 0
            # rule base filter
            line = line.replace(' ', '').strip()
            if len(line) == 0:
                prob_all += 0
                cnt_s += 1
                continue
            counter = collections.Counter(line)
            if counter.most_common(1)[0][1] >= 3:
                prob_all += 0
                cnt_s += 1
                continue
            common_cnt = 0
            last = None
            split_line = list(jieba.cut(line))
            for w in split_line:
                if w == last:
                    common_cnt += 1
                last = w
            if common_cnt >= 2:
                prob_all += 0
                cnt_s += 1
                continue

            # trigram LM
            splitted_line = ['<sos>', '<sos>'] + split_line + ['<eos>', '<eos>']
            for w1, w2, w3 in zip(splitted_line, splitted_line[1:], splitted_line[2:]):
                if (w1, w2) not in model:
                    continue
                else:
                    prob_s += max(model[(w1, w2)][w3], model[(w1, w2)]['BACKGROUND'])
                cnt += 1
            if prob_s == 0:
                prob_all += 0
            else:
                prob_s /= max(cnt, 1)
                prob_all += 10 ** prob_s
            cnt_s += 1
        
        print ('LM score: ' + str(prob_all/cnt_s * 1000))
if __name__ == '__main__':
    main()
