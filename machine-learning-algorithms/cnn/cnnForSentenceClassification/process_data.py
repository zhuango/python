import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(data_folder, clean_string=True):
    """
    Loads data
    """
    revs = []
    train_context_file = data_folder[0]
    train_label_file = data_folder[1]
    test_context_file = data_folder[2]
    test_label_file = data_folder[3]

    trainTag = 0
    testTag = 1

    posTag = "+1"
    negPos = "-1"

    vocab = defaultdict(float)
    with open(train_context_file, "rb") as f:
        train_label = open(train_label_file, "r")
        for line in f:       
            label = train_label.readline().strip();
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            polarity = 0
            if label == posTag:
                polarity = 1;
            else:
                polarity = 0;
            datum  = {"y":polarity, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": trainTag}
                      # "split": np.random.randint(0,cv)}
            revs.append(datum)
        train_label.close()
    with open(test_context_file, "rb") as f:
        test_label = open(test_label_file, "r")
        for line in f:       
            label = test_label.readline().strip();
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            polarity = 0
            if label == posTag:
                polarity = 1;
            else:
                polarity = 0;
            datum  = {"y":polarity, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": testTag}
                      # "split": np.random.randint(0,cv)}
            revs.append(datum)
        test_label.close()
   
    return revs, vocab
    
def get_W(word_vecs, k=100):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        # print("#############################")
        # print(word_vecs[word].shape)
        # print(word)
        # print(W[i].shape)
        # print("#############################")
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def load_vec(fname, vocab):
    """
    format: word vec[50]
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        for line in f:
            strs =line.strip().split(' ')
            if strs[0] in vocab:
                word_vecs[strs[0]] = np.array([float(elem) for elem in strs[1:]], dtype='float32')

    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            # print("************************")
            # print(word)
            # print("************************")
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

import json

if __name__=="__main__":    
    cnnJson = open("process_data.json", "r")
    inputInfo = json.load(cnnJson)
    cnnJson.close()

    TraiContextFile = inputInfo["TraiContext"]
    TestContextFile = inputInfo["TestContext"]
    TraiLabelFile = inputInfo["TraiLabel"]
    TestLabelFile = inputInfo["TestLabel"]
    wordVectorFile = inputInfo["WordVector"]
    outputPath = inputInfo["OutPutPath"]
    mrPath = inputInfo["mrPath"]
    k = 50

    w2v_file = wordVectorFile
    data_folder = [TraiContextFile, TraiLabelFile, TestContextFile, TestLabelFile]    
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder,clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab, k = k)
    W, word_idx_map = get_W(w2v, k)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, k = k)
    W2, _ = get_W(rand_vecs, k)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open(mrPath, "wb"))
    print "dataset created!"
    
