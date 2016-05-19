
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sentence_document_imdb_no_valid
datasets = {'imdb': (sentence_document_imdb_no_valid.load_data, sentence_document_imdb_no_valid.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)
def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of wfhat is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options, dim, sentimentDim):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding

    # randn = numpy.random.rand(options['n_words'],
                              # options['dim_proj'])
    params['Wemb'] = numpy.loadtxt(options['dictPath'], delimiter=' ', dtype='float32')
    # params['Wemb'] = params['Wemb'].astype(config.floatX)
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='lstm_zheng')

    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='lstm_ni')

    # classifier

    # params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
    params['U'] = 0.01 * numpy.random.randn(2*options['dim_proj'],
                                            options['ydim']).astype(config.floatX)

    params['U_zheng'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)

    params['U_ni'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)

    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)


    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        # print(kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = 0.01 * numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U

    # Memory cell Matrix
    V = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'V')] = V

    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):

        preact1 = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact1 += x_

        preact2 = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact2 += tensor.dot(c_, tparams[_p(prefix, 'V')])
        preact2 += x_

        i = tensor.nnet.sigmoid(_slice(preact2, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact2, 1, options['dim_proj']))


        c = tensor.tanh(_slice(preact1, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        preact3 = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact3 += tensor.dot(c, tparams[_p(prefix, 'V')])
        preact3 += x_

        o = tensor.nnet.sigmoid(_slice(preact3, 2, options['dim_proj']))

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}



def adadelta(lr, tparams, grads, x_zheng, x_zheng_mask, x_ni, x_ni_mask, y, cost, cost1, cost2):
    """
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    f_grad_shared_zheng = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask, y], cost1, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared_zheng')

    f_grad_shared_ni = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask, y], cost2, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared_ni')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_grad_shared_zheng, f_grad_shared_ni, f_update

def build_model(alpha, beta, tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x_zheng = tensor.matrix('x_zheng', dtype='int32')
    x_zheng_mask = tensor.matrix('x_zheng_mask', dtype=config.floatX)
    x_ni = tensor.matrix('x_ni', dtype='int32')
    x_ni_mask = tensor.matrix('x_ni_mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int32')

    n_timesteps = x_zheng.shape[0]
    n_samples = x_zheng.shape[1]

    emb_zheng = tparams['Wemb'][x_zheng.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])

    proj1 = get_layer(options['encoder'])[1](tparams, emb_zheng, options,
                                            prefix='lstm_zheng',
                                            mask=x_zheng_mask)
    if options['encoder'] == 'lstm':
        proj_zheng = (proj1 * x_zheng_mask[:, :, None]).sum(axis=0)
        proj_zheng = proj_zheng / x_zheng_mask.sum(axis=0)[:, None]

    emb_ni = tparams['Wemb'][x_ni.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])

    proj2 = get_layer(options['encoder'])[1](tparams, emb_ni, options,
                                            prefix='lstm_ni',
                                            mask=x_ni_mask)

    if options['encoder'] == 'lstm':
        proj_ni = (proj2 * x_ni_mask[:, :, None]).sum(axis=0)
        proj_ni = proj_ni / x_ni_mask.sum(axis=0)[:, None]

    proj = tensor.concatenate((proj_zheng, proj_ni), axis=1)

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    pred_zheng = tensor.nnet.softmax(tensor.dot(proj_zheng, tparams['U_zheng'] + tparams['b']))

    pred_ni = tensor.nnet.softmax(tensor.dot(proj_ni, tparams['U_ni'] + tparams['b']))

    f_pred_prob = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask], pred, name='f_pred_prob')

    f_pred = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask], pred.argmax(axis=1), name='f_pred')

    f_proj = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask], proj, name='f_proj')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost1 = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()
    cost2 = -tensor.log(pred_zheng[tensor.arange(n_samples), y] + off).mean()
    cost3 = -tensor.log(pred_ni[tensor.arange(n_samples), y] + off).mean()
    cost4 = tensor.sum(tensor.square(proj_zheng - proj_ni), axis=1).mean()
    cost = alpha * (cost1 + cost2 + cost3) + beta * cost4

    return use_noise, x_zheng, x_zheng_mask, x_ni, x_ni_mask, y, f_pred_prob, f_pred, cost1, cost2, cost3, cost4, cost, f_proj


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, test_index in iterator:
        x_zheng, x_zheng_mask, x_ni, x_ni_mask, y = prepare_data([data[0][t] for t in test_index],
                                  [data[1][t] for t in test_index],
                                  numpy.array(data[2])[test_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x_zheng, x_zheng_mask, x_ni, x_ni_mask)
        probs[test_index, :] = pred_probs

        n_done += len(test_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x_zheng, x_zheng_mask, x_ni, x_ni_mask, y = prepare_data([data[0][t] for t in valid_index],
                                  [data[1][t] for t in valid_index], numpy.array(data[2])[valid_index],
                                  maxlen=None)
        preds = f_pred(x_zheng, x_zheng_mask, x_ni, x_ni_mask)
        targets = numpy.array(data[2])[valid_index]

        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def process_dict1(file_name, enTrain, enTest, cnTrain, cnTest):
    bTrain1 = 1
    eTrain1 = enTrain+1
    bTest1 = enTrain+1
    eTest1 = enTrain+1+enTest
    bTrain2 = enTrain+1+enTest
    eTrain2 = enTrain+1+enTest+cnTrain
    bTest2 = enTrain+1+enTest+cnTrain
    eTest2 = enTrain+1+enTest+cnTrain+cnTest
    fp = open(file_name, 'r')
    dic1 = {}
    dic2 = {}
    dic3 = {}
    dic4 = {}
    nn = 1
    for line in fp:
        if eTrain1 > nn >= bTrain1:
            if line.strip() in dic1:
                dic1[line.strip()].append(nn)
            else:
                dic1[line.strip()] = [nn]
        elif eTest1 > nn >= bTest1:
            if line.strip() in dic1:
                dic2[nn] = dic1[line.strip()]
            # else:
            # 	  dic2[nn] = None
        elif eTrain2 > nn >= bTrain2:
            if line.strip() in dic3:
                dic3[line.strip()].append(nn)
            else:
                dic3[line.strip()] = [nn]
        elif eTest2 > nn >= bTest2:
            if line.strip() in dic3:
                dic4[nn] = dic3[line.strip()]
        nn += 1
    fp.close()
    return dic2, dic4

def distanceCos(vectorA, vectorB):
    lenA = numpy.sqrt(numpy.dot(vectorA, vectorA))
    lenB = numpy.sqrt(numpy.dot(vectorB, vectorB))
    dotProd = numpy.dot(vectorA, vectorB)
    distance = dotProd / (lenA * lenB)
    return distance

def distanceDot(vectorA, vectorB):
    distance = numpy.dot(vectorA, vectorB)
    return distance
    
def similar_(i, lis, tp, length):
    dic = {}
    minn = numpy.inf
    minIndex = -1
    for j in lis:
        # 1: m = numpy.exp(numpy.dot(tp[i, length:], tp[j, length:]))
        # m = numpy.sqrt(numpy.sum((tp[i-1, length:]-tp[j-1, length:])**2))
        m = distanceCos(tp[i-1, length:], tp[j-1, length:])
        #m = distanceDot(vectorA, vectorB)
        dic[j-1] = m
        if m <= minn:
            minn = m
            minIndex = j-1
    return dic, minIndex

def process_Wemb(dic2, dic4, tparams, length, priorpolarityList):
    tp = tparams['Wemb'].get_value()
    # aaa = len(dic2)+len(dic4)
    # nn = 0
    for i in dic2:
        # nn += 1
        # if nn % 1000 == 0:
        #     edd = time.time()
        #     print nn, " of ", aaa, edd
        if(priorpolarityList[i]):
            index_dic, min_index = similar_(i, dic2[i], tp, length)
        # 1: tp[i, length:] = tp[min_index, length:]
            tp[i-1, :length] = tp[min_index, :length]
        """
        tp[i, :length] = 0.0
        for ii in index_dic:
            tp[i, :length] += tp[ii, :length]*index_dic[ii]
        """
    for j in dic4:
        # nn += 1
        # if nn % 1000 == 0:
        #     edd = time.time()
        #     print nn, " of ", aaa, edd
        if(priorpolarityList[i]):
            index_dic1, min_index1 = similar_(j, dic4[j], tp, length)
        # 1: tp[j, length:] = tp[min_index1, length:]
            tp[j-1, :length] = tp[min_index1, :length]
        """
        tp[j, :length] = 0.0
        for jj in index_dic1:
            tp[j, :length] += tp[jj, :length]*index_dic1[jj]
        """
    tparams['Wemb'].set_value(tp)

def get_proj(f_proj, prepare_data, data, iterator, dim_proj, verbose=False):
    """
    Get the top hidden layer
    """
    n_samples = len(data[0])
    projs = numpy.zeros((n_samples, 2*dim_proj)).astype(config.floatX)
    # projs = numpy.zeros((n_samples, dim_proj)).astype(config.floatX)

    for _, index in iterator:
        x_zheng, x_zheng_mask, x_ni, x_ni_mask, y = prepare_data([data[0][t] for t in index],
                                  [data[1][t] for t in index],
                                  numpy.array(data[2])[index],
                                  maxlen=None)
        hidden_projs = f_proj(x_zheng, x_zheng_mask, x_ni, x_ni_mask)
        projs[index, :] = hidden_projs

    return projs


def train_lstm(
    type='',
    dim_proj=-1,  # word embeding dimension and LSTM number of hidden units.
    sentimentDim=-1,
    patience=50,  # Number of epoch to wait before early stop if no progress
    category='',
    dataSetPath = "",
    dictPath = "",
    max_epochs=-1,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.1,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=-1,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    validFreq=400,  # Compute the validation error after this number of update.
    saveFreq=400,  # Save the parameters after every saveFreq updates
    maxlen=-1,  # Sequence longer then this get ignored
    batch_size=10,  # The batch size during training.
    test_batch_size=1,# The batch size used for test set.
    alpha=1,
    beta=1,
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
    test_size=-1,  # If >0, we keep only this number of test example.
):
    # reload_model='lstm_model_'+category+'.npz',  # Path to a saved model we want to start from.
    reload_model = None  # Path to a saved model we want to start from.
    saveto=dataSetPath + 'lstm_model_'+category+'.npz'
    # Model options
    embedName = category+'_'+str(dim_proj)
    model_options = locals().copy()

    model_options['dictPath'] = dictPath
    
    enTrain = wcWord(os.path.dirname(dictPath) +"/" + type+"_train_"+category+"_en_"+str(dim_proj - sentimentDim)+".txt")
    cnTrain = wcWord(os.path.dirname(dictPath)+"/" + type+"_train_"+category+"_cn_"+str(dim_proj - sentimentDim)+".txt")
    enTest = wcWord(os.path.dirname(dictPath)+"/" + type+"_test_"+category+"_en_"+str(dim_proj - sentimentDim)+".txt")
    cnTest = wcWord(os.path.dirname(dictPath)+"/" + type+"_test_"+category+"_cn_"+str(dim_proj - sentimentDim)+".txt")
    wordVectorDim = dim_proj - sentimentDim
    priorpolarityList = loadPriorpolarityPosList(os.path.dirname(dictPath)+"/" + category +'_wordList.txt')
    
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, test = load_data(path=dataSetPath + category+'.pkl', n_words=n_words, maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        # numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx], [test[2][n] for n in idx])

    ydim = numpy.max(train[2]) + 1

    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, dim_proj, sentimentDim)

    if reload_model:
        load_params(saveto, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)
    # print("---------------------------------------------------")
    # print(tparams['Wemb'].get_value().shape), 335388+146881+369464+159125
    # use_noise is for dropout
    (use_noise, x_zheng, x_zheng_mask, x_ni, x_ni_mask, y, f_pred_prob, f_pred,
     cost1, cost2, cost3, cost4, cost, f_proj) = build_model(alpha, beta, tparams, model_options)


    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += tensor.sqrt((tparams[_p('lstm_zheng', 'W')] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm_zheng', 'U')] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm_zheng', 'V')] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm_ni', 'W')] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm_ni', 'U')] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm_ni', 'V')] ** 2).sum())
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x_zheng, x_zheng_mask, x_ni, x_ni_mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_grad_shared_zheng, f_grad_shared_ni, f_update \
        = optimizer(lr, tparams, grads, x_zheng, x_zheng_mask, x_ni, x_ni_mask, y, cost, cost1, cost2)

    print 'Optimization'

    kf_test = get_minibatches_idx(len(test[0]), test_batch_size)

    print "%d train examples" % len(train[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done

    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)

    estop = False  # early stop
    start_time = time.time()

    try:
        updateTime = 0

        f = open(dataSetPath+'/record.txt', 'w')

        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=False)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[2][t] for t in train_index]
                x_ni = [train[1][t]for t in train_index]
                x_zheng = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x1, x1_mask, x2, x2_mask, y = prepare_data(x_zheng, x_ni, y)
                n_samples += x1.shape[1]

                cost = f_grad_shared(x1, x1_mask, x2, x2_mask, y)
                cost1 = f_grad_shared_zheng(x1, x1_mask, x2, x2_mask, y)
                cost2 = f_grad_shared_ni(x1, x1_mask, x2, x2_mask, y)

                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Cost1', cost1, 'Cost2', cost2

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p

                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
                    #process_dict1(file_name, enTrain, enTest, cnTrain, cnTest)
                    afd, acd = process_dict1(os.path.dirname(dictPath)+"/" + category +'_wordList.txt', enTrain, cnTest, cnTrain, cnTest)
                    process_Wemb(afd, acd, tparams,wordVectorDim, priorpolarityList)
                    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    test_proj = get_proj(f_proj, prepare_data, test, kf_test, dim_proj)
                    train_proj = get_proj(f_proj, prepare_data, train, kf, dim_proj)

                    history_errs.append([test_err])

                    print('Accuracy:' + str(1-float(test_err)))
                    f.write('Accuracy:' + str(1-float(test_err)))
                    f.write('\n')

                    if (uidx == 0 or
                        test_err <= numpy.array(history_errs)[:].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                        test_prob_best = pred_probs(f_pred_prob, prepare_data, test, kf_test)

                        # numpy.savetxt('results/'+embedName+'_embedding'+str(dim_proj)+'/train_proj_best.txt', train_proj, fmt='%.4f', delimiter=' ')
                        # numpy.savetxt('paper experiment/'+embedName+'/test_proj_best.txt', test_proj, fmt='%.4f', delimiter=' ')
                        numpy.savetxt(dataSetPath+'test_best.txt', test_prob_best, fmt='%.2f', delimiter=' ')
                        numpy.savetxt(dataSetPath+'embeddings_best.txt', params['Wemb'], fmt='%.4f', delimiter=' ')

                    print ('Train ', train_err, 'Test ', test_err)

                    if (len(history_errs) > patience and
                        test_err >= numpy.array(history_errs)[:].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            # numpy.savetxt(embedName+'/embeddings_'+str(eidx)+'.txt', params['Wemb'], fmt='%.4f', delimiter=' ')

            train_prob = pred_probs(f_pred_prob, prepare_data, train, kf_train_sorted)
            test_prob = pred_probs(f_pred_prob, prepare_data, test, kf_test)

            numpy.savetxt(dataSetPath+'/test_proj_'+str(eidx)+'.txt', test_proj, fmt='%.4f', delimiter=' ')
            numpy.savetxt(dataSetPath+'/train_proj_'+str(eidx)+'.txt', train_proj, fmt='%.4f', delimiter=' ')

            # numpy.savetxt('paper experiment/'+embedName+'/train_prob_'+str(eidx)+'.txt', train_prob, fmt='%.2f', delimiter=' ')
            numpy.savetxt(dataSetPath+'/test_prob_'+str(eidx)+'.txt', test_prob, fmt='%.2f', delimiter=' ')

            print 'Seen %d samples' % n_samples

            if estop:
                break

        f.close()
    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    afd, acd = process_dict1(os.path.dirname(dictPath) + "/"+ category +'_wordList.txt', enTrain, cnTest, cnTrain, cnTest)
    process_Wemb(afd, acd, tparams, wordVectorDim,priorpolarityList)
    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err, test_err=test_err,
                    history_errs=history_errs, **best_p)

    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, test_err

def preprocess(type, category, dimension, dataPath = "G:/liuzhuang/corpus/Serializer/", dataSetPath = ""):
    import cPickle
    print("preprocessing...")
    train_data_x_zheng = []
    train_data_x_ni = []
    train_data_y = []
    test_data_x_zheng = []
    test_data_x_ni = []
    test_data_y = []
    train_vec_file1 = open(dataPath+type+'_train_'+category+'_en_'+str(dimension)+'.txt', 'r')
    train_vec_file2 = open(dataPath+type+'_train_'+category+'_cn_'+str(dimension)+'.txt', 'r')
    train_label_file = open(dataPath + 'train_'+category+'_label.txt', 'r')
    for i in range(0, 4000):
        vec_line1 = train_vec_file1.readline()
        vec_line2 = train_vec_file2.readline()

        label_line = train_label_file.readline()
        train_data_x_zheng.append([int(elem) for elem in vec_line1.split(' ')])
        train_data_x_ni.append([int(elem) for elem in vec_line2.split(' ')])
        if label_line[0] == '1':
            train_data_y.extend([1])
        else:
            train_data_y.extend([0])

    test_vec_file1 = open(dataPath+type+'_test_'+category+'_en_'+str(dimension)+'.txt', 'r')
    test_vec_file2 = open(dataPath+type+'_test_'+category+'_cn_'+str(dimension)+'.txt', 'r')
    test_label_file = open(dataPath + 'test_'+category+'_label.txt', 'r')
    for i in range(0, 4000):
        vec_line1 = test_vec_file1.readline()
        vec_line2 = test_vec_file2.readline()
        label_line = test_label_file.readline()
        test_data_x_zheng.append([int(elem) for elem in vec_line1.split(' ')])
        test_data_x_ni.append([int(elem) for elem in vec_line2.split(' ')])
        if label_line[0] == '1':
            test_data_y.extend([1])
        else:
            test_data_y.extend([0])

    output_file = open(dataSetPath+category+'.pkl', 'w')

    train_data = [train_data_x_zheng, train_data_x_ni, train_data_y]
    test_data = [test_data_x_zheng, test_data_x_ni, test_data_y]
    cPickle.dump(train_data, output_file)
    cPickle.dump(test_data, output_file)

    output_file.close()

def WordCount(SeriPath, type, category, dimension):
    #######################
    filenames = []
    filenames.append(SeriPath + type+"_test_"+category+"_en_"+str(dimension)+".txt")
    filenames.append(SeriPath + type+"_test_"+category+"_cn_"+str(dimension)+".txt")
    filenames.append(SeriPath + type+"_train_"+category+"_en_"+str(dimension)+".txt")
    filenames.append(SeriPath + type+"_train_"+category+"_cn_"+str(dimension)+".txt")
    wordsnumber=0
    maxLen = 0
    for filename in filenames:
        wordsnumber += wcWord(filename)
        tmpLen = maxWordLen(filename)
        if(maxLen < maxWordLen(filename)):
            maxLen = tmpLen
    print(wordsnumber)
    return wordsnumber, maxLen
    #######################

import os
from wc import wcWord, maxWordLen
from loadDict import loadPriorpolarityPosList
import json 

if __name__ == '__main__':

    f = open('sflLSTM.json', 'r')
    inputInfo = json.load(f)
    f.close()
    
    # See function train for all possible parameter and there definition.
    category = 'book'
    dimension = 100
    sentimentDim = 50
    wordDimension = dimension - sentimentDim
    type = 'semantic_sentiment'
    # type = 'semantic'
    outputDir = inputInfo["TotalOutputDir"] + str(wordDimension) + "d/" + category + "/"
    SeriPath = inputInfo["SerializerDir"]
    dictPath = SeriPath + type + "_" + category +"_dict_"+ str(wordDimension)+".txt"
    if(not os.path.exists(outputDir)):
        os.makedirs(outputDir)

    preprocess(type, category, wordDimension,SeriPath, outputDir)
    
    n_words,maxlen = WordCount(SeriPath, type, category, wordDimension);
    train_lstm(
        type=type,
        dim_proj=dimension,
        sentimentDim=sentimentDim,
        category=category,
        dataSetPath=outputDir,
        dictPath = dictPath,
        max_epochs=10,
        n_words=n_words,
        maxlen=maxlen + 1,
        decay_c=0.0001,
        test_size=4000,
        alpha=1,
        beta=1,
    )
"""
if __name__ == "__main__":
    afd, acd = process_dict1('music_wordList.txt', 335388, 146881, 369464, 159125)
    print len(afd), len(acd)
    Wemb = numpy.loadtxt("semantic_sentiment_music_dict_100.txt")
    Wemb_ = theano.shared(Wemb, name="Wemb_")
    tt = {"Wemb": Wemb_}
    print Wemb.shape
    print "--------------------------------------"
    print numpy.sqrt(numpy.sum((Wemb[222, 100:]-Wemb[333, 100:])**2))
    process_Wemb(afd, acd, tt, 100)
    print "--------------------------------------"
"""