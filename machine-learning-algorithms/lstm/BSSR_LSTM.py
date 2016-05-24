
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb_no_valid

datasets = {'imdb': (imdb_no_valid.load_data, imdb_no_valid.prepare_data)}


from wc import wc
from writeNonsenceLabel import writeNonsenceLabel
from writeNonsenceVector import writeNonsenceVector
from genSentenceVector import genSentenceVector
from AddZerosVectorToSent import AddZerosVectorToSent

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
        # Make a minibatch out of what is left
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


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding

    # randn = numpy.random.rand(options['n_words'],
    #                           options['dim_proj'])
    #params['Wemb'] = numpy.loadtxt("paper experiment/dev_seq_relation_new_200/data/embeddings200_seq_relation_0119.txt", delimiter=' ')
    params['Wemb'] = numpy.loadtxt(options['dictPath'], delimiter=' ',dtype='float32')
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
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
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
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



def adadelta(lr, tparams, grads, x, mask, y, cost):
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

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

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

    return f_grad_shared, f_update

def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int32')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    f_proj = theano.function([x, mask], proj, name='f_proj')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, f_proj


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, test_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in test_index],
                                  numpy.array(data[1])[test_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
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
        x, x_mask, labels = prepare_data([data[0][t] for t in valid_index],
                                  [data[1][t] for t in valid_index],
                                  maxlen=None)
        preds = f_pred(x, x_mask)
        targets = numpy.array(data[1])[valid_index]

        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def get_proj(f_proj, prepare_data, data, iterator, dim_proj, verbose=False):
    """
    Get the top hidden layer
    """
    n_samples = len(data[0])
    projs = numpy.zeros((n_samples, dim_proj)).astype(config.floatX)

    for _, index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in index],
                                  numpy.array(data[1])[index],
                                  maxlen=None)
        hidden_projs = f_proj(x, mask)
        projs[index, :] = hidden_projs

    return projs

import os
def train_lstm(
    lstmOutPutRootPath = "",
    dictPath = "",
    dataSetPath= "",
    dim_proj=200,  # word embeding dimension and LSTM number of hidden units.
    patience=50,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=6284,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=200,  # Compute the validation error after this number of update.
    saveFreq=200,  # Save the
    #  parameters after every saveFreq updates
    maxlen=137,  # Sequence longer then this get ignored
    batch_size=10,  # The batch size during training.
    valid_batch_size=5,  # The batch size used for validation/test set.
    test_batch_size=1,# The batch size used for test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    #embedName='dev_seq_relation_new_200',
):
    errorThreshold = 0.0009
    countLessThanThreshold = 0
    exitThresholdHitCount = 7
    # Model options
    model_options = locals().copy()
    model_options['dictPath'] = dictPath
    print "model options", model_options
    
    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, test = load_data(path = dataSetPath, n_words=n_words, maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        # numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim
    
    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, f_proj) = build_model(tparams, model_options)


    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += tensor.sqrt((tparams['U'] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm', 'W')] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm', 'U')] ** 2).sum())
        weight_decay += tensor.sqrt((tparams[_p('lstm', 'V')] ** 2).sum())
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

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

        f = open(lstmOutPutRootPath+'/record.txt', 'w')

        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=False)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

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
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    test_proj = get_proj(f_proj, prepare_data, test, kf_test, dim_proj)
                    train_proj = get_proj(f_proj, prepare_data, train, kf, dim_proj)

                    history_errs.append([test_err])

                    f.write('Accuracy:' + str(1-float(test_err)))
                    f.write('\n')

                    if (uidx == 0 or
                        test_err <= numpy.array(history_errs)[:].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                        test_prob_best = pred_probs(f_pred_prob, prepare_data, test, kf_test)

                        # numpy.savetxt('results/'+embedName+'_embedding'+str(dim_proj)+'/train_proj_best.txt', train_proj, fmt='%.4f', delimiter=' ')
                        numpy.savetxt(lstmOutPutRootPath+'/test_proj_best.txt', test_proj, fmt='%.4f', delimiter=' ')
                        #numpy.savetxt('paper experiment/'+embedName+'/test_best.txt', test_prob_best, fmt='%.2f', delimiter=' ')
                        #numpy.savetxt('paper experiment/'+embedName+'/embeddings_best.txt', params["Wemb"], fmt='%.4f', delimiter=' ')

                    print ('Train ', train_err, 'Test ', test_err)
                    if(train_err < errorThreshold):
                        countLessThanThreshold += 1
                        if(countLessThanThreshold > exitThresholdHitCount):
                            estop = True

                    if (len(history_errs) > patience and
                        test_err >= numpy.array(history_errs)[:].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            numpy.savetxt(lstmOutPutRootPath+'/embeddings_'+str(eidx)+'.txt', params["Wemb"], fmt='%.4f', delimiter=' ')

            train_prob = pred_probs(f_pred_prob, prepare_data, train, kf_train_sorted)
            test_prob = pred_probs(f_pred_prob, prepare_data, test, kf_test)

            numpy.savetxt(lstmOutPutRootPath+'/test_proj_'+str(eidx)+'.txt', test_proj, fmt='%.4f', delimiter=' ')
            # numpy.savetxt('results/'+embedName+'_embedding'+str(dim_proj)+'/train_proj_'+str(eidx)+'.txt', train_proj, fmt='%.4f', delimiter=' ')

            numpy.savetxt(lstmOutPutRootPath+'/train_prob_'+str(eidx)+'.txt', train_prob, fmt='%.2f', delimiter=' ')
            numpy.savetxt(lstmOutPutRootPath+'/test_prob_'+str(eidx)+'.txt', test_prob, fmt='%.2f', delimiter=' ')

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


def AddSeriCountToTimesOfBatch_size(fileName, batch_size, wordDimension, randHigh):
    count = wc(fileName)
    countNeedToAdd = (batch_size - (count % batch_size))%batch_size
    writeNonsenceVector(fileName, countNeedToAdd, wordDimension, randHigh)
    
    
def AddLabelCountToTimesOfBatch_size(fileName, batch_size):
    count = wc(fileName)
    countNeedToAdd = (batch_size - (count % batch_size))%batch_size
    writeNonsenceLabel(fileName, countNeedToAdd)

#def preprocess(type, category, dimension, dataSetPath = ""):
def preprocess(seriFilePath, labelFilePath, dataSetPath, wordDimension, batch_size = 10,dictCount = 1000):
#def preprocess(seriFilePath_train = seriFilePath_train, 
#               seriFilePath_test = seriFilePath_test,
#               labelFilePath_train = labelFilePath_train, 
#               labelFilePath_test = labelFilePath_test, 
#               dataSetPath = datasetPath, 
#               wordDimension = wordDimension, 
#               batch_size = 200)
    print('Converting data format...')
    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []
    # train_vec_file = open("H:/CNN_YANG/"+embedName+"/train_"+str(wordDimension+posDimension)+".txt", 'r')
    # train_label_file = open("H:/CNN_YANG/"+embedName+"/train_label_"+str(wordDimension+posDimension)+".txt", 'r')
    
    #############################################
    AddSeriCountToTimesOfBatch_size(seriFilePath, batch_size, wordDimension, dictCount)
    AddLabelCountToTimesOfBatch_size(labelFilePath, batch_size)
    #############################################
    train_vec_file=open(seriFilePath, 'r')
    train_label_file = open(labelFilePath, 'r')
    # train_vec_file=open("yuliao/book/outEmbedding_Trim.txt", 'r')
    # train_label_file = open("yuliao/book/dvd_label.txt", 'r')
    vectorCount = wc(seriFilePath)
    
    for i in range(0, vectorCount):
        vec_line = train_vec_file.readline().strip()

        label_line = train_label_file.readline().strip()
        train_data_x.append([float(elem) for elem in vec_line.split(' ')])
        if label_line[0] == '1':
            train_data_y.extend([1])
        else:
            train_data_y.extend([0])

    test_vec_file = open(seriFilePath, 'r')
    test_label_file = open(labelFilePath, 'r')

    for i in range(0, vectorCount):
        vec_line = test_vec_file.readline().strip()
        label_line = test_label_file.readline().strip()
        test_data_x.append([float(elem) for elem in vec_line.split(' ')])
        if label_line[0] == '1':
            test_data_y.extend([1])
        else:
            test_data_y.extend([0])

    output_file = open(dataSetPath, 'wb')

    train_data = [train_data_x, train_data_y]
    test_data = [test_data_x, test_data_y]
    pkl.dump(train_data, output_file)
    pkl.dump(test_data, output_file)

    output_file.close()

def SingleProcess(corpusRootPath, lstmOutputRootPath, clas, language, wordDimension, corpusType):
    
    representationDim = 50
    corpusPath = corpusRootPath
    lstmOutputPath = lstmOutputRootPath + corpusType+ "/"
    
    branchPath = str(wordDimension)+"d/"+language+"/"+clas+"/"
    if(not os.path.exists(lstmOutputPath + branchPath)):
        os.makedirs(lstmOutputPath + branchPath)
    dictPath = corpusPath + language + "/"+corpusType+"_"+clas+"_new.txt.extract_"+str(wordDimension)+".lstmDict"
    seriFilePath = corpusPath + language + "/"+corpusType+"_"+clas+"_new.txt.extract_"+str(wordDimension)+".serialization"
    labelFilePath = corpusPath + language + "/"+corpusType+"_"+clas+"_new.txt.label"
    datasetPath = lstmOutputPath + branchPath + clas +"_dataSet"+str(wordDimension)+".pkl"
                
    if(not os.path.exists(datasetPath)):
        preprocess(seriFilePath, labelFilePath,datasetPath, wordDimension, batch_size = 10, dictCount = wc(dictPath))
    if(not os.path.exists(lstmOutputPath + branchPath+'/test_proj_best.txt')):
        train_lstm(
            lstmOutPutRootPath = lstmOutputPath + branchPath,
            dictPath = dictPath,
            dataSetPath = datasetPath,
            dim_proj = wordDimension,
            n_words = wc(dictPath),
            max_epochs=30,
            test_size=wc(seriFilePath)
            )
    numberFile = corpusPath+language+"/"+corpusType+"_"+clas+"_new.txt.number"
    fragmentVectorFile = lstmOutputPath+str(wordDimension)+"d/"+language+"/"+clas+"/test_proj_best.txt"
    indexFile = lstmOutputPath+str(wordDimension)+"d/"+language+"/"+clas+"/" + ""+corpusType+"_"+clas+"_new.txt.index"
    sentenceVectorFile = lstmOutputPath+str(wordDimension)+"d/"+language+"/"+clas+"/" + ""+corpusType+"_"+clas+"_new.txt.sent"
    genSentenceVector(numberFile, fragmentVectorFile, indexFile, sentenceVectorFile, representationDim)
    
    branchPath = str(wordDimension)+"d/"+language+"/"+clas+"/"
    indexFile = lstmOutputPath + branchPath + corpusType+"_"+clas+"_new.txt.index"
    sentFile = lstmOutputPath + branchPath + corpusType+"_"+clas+"_new.txt.sent"
    numberFile = corpusPath + language + "/"+corpusType+"_"+clas+"_new.txt.number"
    newSentFile = lstmOutputPath + branchPath +clas+"_"+corpusType+"_embed_" +str.upper(language)+".sent"
    newindexFile = lstmOutputPath + branchPath +clas+"_"+corpusType+"_index_" +str.upper(language)+".sent"
    AddZerosVectorToSent(indexFile, sentFile, numberFile, newSentFile, newindexFile, representationDim)
                
from multiprocessing import Process
import json

if __name__ == '__main__':

    f = open('BSSR.json', 'r')
    inputInfo = json.load(f)
    f.close()

    corpusRootPath = inputInfo['CorpusPath']
    lstmOutputRootPath = inputInfo['LstmOutputPath']

    # See function train for all possible parameter and there definition.
    category = 'dvd'
    dimension = 300
    sentimentDim = 0
    # type = 'semantic_sentiment'
    type = 'semantic'
    
    classes = ["music"]#, "book", "dvd"
    languages = ["en", "cn"]
    wordDimensions = [50]#, 100
    corpusTypes = ["label", "test"]
    processCount = 0
    for corpusType in corpusTypes:
        for clas in classes:
            for language in languages:
                for wordDimension in wordDimensions:
                    processCount += 1
                    #SingleProcess(clas, language, wordDimension)#
                    p = Process(target=SingleProcess, args=(corpusRootPath, lstmOutputRootPath, clas, language, wordDimension,corpusType))
                    p.start()
                    print(str(wordDimension) + " " + language + " " + clas + " is running. PID: " + str(p.ident))
                    #if(processCount % 3 == 0):  p.join()
                    p.join() # one by one.
