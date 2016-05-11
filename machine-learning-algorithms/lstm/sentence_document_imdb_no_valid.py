import cPickle

import numpy
import theano


def prepare_data(seqs_zheng, seqs_ni, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths_zheng = [len(s) for s in seqs_zheng]
    lengths_ni = [len(s) for s in seqs_ni]

    if maxlen is not None:
        new_seqs_zheng = []
        new_seqs_ni = []
        new_labels = []
        new_zheng_lengths = []
        new_ni_lengths = []
        for l_zheng, l_ni, s1, s2, y in zip(lengths_zheng, lengths_ni, seqs_zheng, seqs_ni, labels):
            if l_zheng < maxlen and l_ni < maxlen:
                new_seqs_zheng.append(s1)
                new_seqs_ni.append(s2)
                new_labels.append(y)
                new_zheng_lengths.append(l_zheng)
                new_ni_lengths.append(l_ni)
        lengths_zheng = new_zheng_lengths
        lengths_ni = new_ni_lengths
        labels = new_labels
        seqs_zheng = new_seqs_zheng
        seqs_ni = new_seqs_ni

        if len(lengths_zheng) < 1 or len(lengths_ni) < 1:
            return None, None, None

    n_samples = len(seqs_zheng)
    lengths = lengths_zheng + lengths_ni

    maxlen = numpy.max(lengths)

    x_zheng = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_zheng_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    x_ni = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_ni_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for idx, s1 in enumerate(seqs_zheng):
        x_zheng[:lengths_zheng[idx], idx] = s1
        x_zheng_mask[:lengths_zheng[idx], idx] = 1.

    for idx, s2 in enumerate(seqs_ni):
        x_ni[:lengths_ni[idx], idx] = s2
        x_ni_mask[:lengths_ni[idx], idx] = 1.

    return x_zheng, x_zheng_mask, x_ni, x_ni_mask, labels



def load_data(path='', n_words=4000, maxlen=None,
              sort_by_len=True):

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    f = open(path, 'rb')

    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x1 = []
        new_train_set_x2 = []
        new_train_set_y = []
        for x1, x2, y in zip(train_set[0], train_set[1], train_set[2]):
            if len(x1) < maxlen:
                new_train_set_x1.append(x1)
                new_train_set_x2.append(x2)
                # print(x)
                new_train_set_y.append(y)
                # print(y)
        train_set = (new_train_set_x1, new_train_set_x2, new_train_set_y)
        del new_train_set_x1, new_train_set_x2, new_train_set_y

    train_set_x1, train_set_x2, train_set_y = train_set
    n_samples = len(train_set_x1)
    print(n_samples)
    sidx = numpy.random.permutation(n_samples)
    n_train = n_samples
    train_set_x1 = [train_set_x1[s] for s in sidx[:n_train]]
    train_set_x2 = [train_set_x2[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x1, train_set_x2, train_set_y)


    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x1, test_set_x2, test_set_y = test_set
    train_set_x1, train_set_x2, train_set_y = train_set

    train_set_x1 = remove_unk(train_set_x1)
    train_set_x2 = remove_unk(train_set_x2)
    test_set_x1 = remove_unk(test_set_x1)
    test_set_x2 = remove_unk(test_set_x2)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(train_set_x1)
        train_set_x1 = [train_set_x1[i] for i in sorted_index]
        train_set_x2 = [train_set_x2[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x1, train_set_x2, train_set_y)
    test = (test_set_x1, test_set_x2, test_set_y)

    return train, test

