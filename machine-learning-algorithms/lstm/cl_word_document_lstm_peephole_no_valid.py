
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
    #                           options['dim_proj'])
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
        print(kk)
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
    validFreq=100,  # Compute the validation error after this number of update.
    saveFreq=200,  # Save the parameters after every saveFreq updates
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
    errorThreshold = 0.009
    countLessThanThreshold = 0
    exitThresholdHitCount = 5
    diffStr = " (" + category + " " +str(dim_proj) + "d " + str(os.getpid()) +" " + str(alpha) +"_"+str(beta) +")"
    # reload_model='lstm_model_'+category+'.npz',  # Path to a saved model we want to start from.
    reload_model = None  # Path to a saved model we want to start from.
    saveto=dataSetPath + 'lstm_model_'+category+'.npz'
    # Model options
    embedName = 'thesis_experiment_'+category+'_'+str(dim_proj)
    model_options = locals().copy()

    model_options['dictPath'] = dictPath

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
                # print x1.shape[0]
                # print x2.shape[0]
                cost = f_grad_shared(x1, x1_mask, x2, x2_mask, y)
                cost1 = f_grad_shared_zheng(x1, x1_mask, x2, x2_mask, y)
                cost2 = f_grad_shared_ni(x1, x1_mask, x2, x2_mask, y)

                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Cost1', cost1, 'Cost2', cost2, diffStr

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
                    print('11111111111' + diffStr)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    print('22222222222'+diffStr)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    print('33333333333' +diffStr)
                    # test_proj = get_proj(f_proj, prepare_data, test, kf_test, dim_proj)
                    # print('44444444444')
                    # train_proj = get_proj(f_proj, prepare_data, train, kf, dim_proj)
                    # print('55555555555')
                    history_errs.append([test_err])
                    print('4444444444' + diffStr)
                    print('Accuracy:' + str(1-float(test_err)) + diffStr)
                    f.write('Accuracy:' + str(1-float(test_err)))
                    f.write('\n')

                    if (uidx == 0 or
                        test_err <= numpy.array(history_errs)[:].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                        test_prob_best = pred_probs(f_pred_prob, prepare_data, test, kf_test)

                        # numpy.savetxt('results/'+embedName+'_embedding'+str(dim_proj)+'/train_proj_best.txt', train_proj, fmt='%.4f', delimiter=' ')
                        # numpy.savetxt('paper experiment/'+embedName+'/test_proj_best.txt', test_proj, fmt='%.4f', delimiter=' ')
                        numpy.savetxt(dataSetPath+'/test_best.txt', test_prob_best, fmt='%.4f', delimiter=' ')
                        # numpy.savetxt(embedName+'/embeddings_best.txt', params['Wemb'], fmt='%.4f', delimiter=' ')

                    print ('Train ', train_err, 'Test ', test_err, diffStr)
                    # if(train_err < errorThreshold):
                    #     countLessThanThreshold += 1
                    #     if(countLessThanThreshold > exitThresholdHitCount):
                    #         estop = True

            # numpy.savetxt(embedName+'/embeddings_'+str(eidx)+'.txt', params['Wemb'], fmt='%.4f', delimiter=' ')

            # train_prob = pred_probs(f_pred_prob, prepare_data, train, kf_train_sorted)
            test_prob = pred_probs(f_pred_prob, prepare_data, test, kf_test)

            # numpy.savetxt(embedName+'/test_proj_'+str(eidx)+'.txt', test_proj, fmt='%.4f', delimiter=' ')
            # numpy.savetxt(embedName+'/train_proj_'+str(eidx)+'.txt', train_proj, fmt='%.4f', delimiter=' ')

            # numpy.savetxt('paper experiment/'+embedName+'/train_prob_'+str(eidx)+'.txt', train_prob, fmt='%.2f', delimiter=' ')
            numpy.savetxt(dataSetPath+'/test_prob_'+str(eidx)+'.txt', test_prob, fmt='%.4f', delimiter=' ')

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

    print 'Train ', train_err, 'Test ', test_err, diffStr
    if saveto:
        numpy.savez(saveto, train_err=train_err, test_err=test_err,
                    history_errs=history_errs, **best_p)

    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))), diffStr
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
    train_vec_file1 = open(dataPath+type+'_train_'+category+'_en.txt', 'r')
    train_vec_file2 = open(dataPath+type+'_train_'+category+'_cn.txt', 'r')
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

    test_vec_file1 = open(dataPath+type+'_test_'+category+'_en.txt', 'r')
    test_vec_file2 = open(dataPath+type+'_test_'+category+'_cn.txt', 'r')
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

    output_file = open(dataSetPath + category+'.pkl', 'w')

    train_data = [train_data_x_zheng, train_data_x_ni, train_data_y]
    test_data = [test_data_x_zheng, test_data_x_ni, test_data_y]
    cPickle.dump(train_data, output_file)
    cPickle.dump(test_data, output_file)

    output_file.close()

def WordCount(SeriPath, type, category, dimension):
    #######################
    filenames = []
    filenames.append(SeriPath + type+"_test_"+category+"_en.txt")
    filenames.append(SeriPath + type+"_test_"+category+"_cn.txt")
    filenames.append(SeriPath + type+"_train_"+category+"_en.txt")
    filenames.append(SeriPath + type+"_train_"+category+"_cn.txt")
    wordsnumber=0
    maxLen = 0
    wordsnumber = wc(SeriPath +type + "_"+category+"_dict_" +str(dimension) + ".txt")
    for filename in filenames:
        tmpLen = maxWordLen(filename)
        if(maxLen < tmpLen):
            maxLen = tmpLen
    print(wordsnumber)
    return wordsnumber, maxLen
    #######################

def SingleProcess(
    type, 
    category,
    dimension, 
    sentimentDim, 
    SerializerDir, 
    TotalOutputDir, 
    Alpha, 
    Beta
):
    
    wordDimension = dimension - sentimentDim
    
    outputDir = TotalOutputDir + str(wordDimension) + "d/" + category +"_"+str(Alpha)+"_"+str(Beta)+"/"
    SeriPath = SerializerDir
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
        max_epochs=15,
        n_words=n_words,
        maxlen=maxlen + 1,
        decay_c=0.001,
        test_size=4000,
        alpha=Alpha,#2, 4, 6, 8, 10
        beta=Beta
    )

import os
from wc import wcWord, maxWordLen, wc
import json
from multiprocessing import Process
import subprocess
import re
def findProcess( processId ):
    ps= subprocess.Popen("ps -ef | grep "+str(processId), shell=True, stdout=subprocess.PIPE)
    #ps= subprocess.Popen(r'tasklist.exe /NH /FI "PID eq %s"' % str(processId), shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()
    return str(output)
def isProcessRunning( processId):
    output = findProcess( processId )
    if re.search(" "+str(processId) +" ", output) is None:
        return False
    else:
        return True
        
if __name__ == '__main__':

    f = open('clLSTM.json', 'r')
    inputInfo = json.load(f)
    f.close()
    
    # See function train for all possible parameter and there definition.
    categories = ['dvd', 'music', 'book']#, 'dvd', 'music'
    dimensions = [150] # 100, 150, 250
    sentimentDim = 50
    type = 'semantic_sentiment'
    #type = 'semantic'
    alphaes = [1, 2, 4, 6, 8, 10]
    
    TotalOutputDir = inputInfo["TotalOutputDir"]
    SerializerDir = inputInfo["SerializerDir"]

    pid =22159
    while(isProcessRunning(pid)):
        print(str(pid) + " is running.\n")
        time.sleep(5 * 60)

    for dimension in dimensions:
        for category in categories:
            SingleProcess(type,category,150,sentimentDim,SerializerDir,TotalOutputDir,3,7)
            SingleProcess(type,category,150,sentimentDim,SerializerDir,TotalOutputDir,7,3)
    #         argsForProcess = (type,category,dimension,sentimentDim,SerializerDir,TotalOutputDir,9,1)
    #         p = Process(target=SingleProcess, args=argsForProcess)
    #         p.start()
    #         print(category + " is running. PID: " + str(p.ident))
    #         p.join()# one by one.

    #SingleProcess(type,"book",100,sentimentDim,SerializerDir,TotalOutputDir+"book9_1/",9,1)
    
    #SingleProcess(type,"book",150,sentimentDim,SerializerDir,TotalOutputDir+"book9_1/",9,1)
    
    # SingleProcess(type,"dvd",150,sentimentDim,SerializerDir,TotalOutputDir,2,8)
    
    # SingleProcess(type,"music",150,sentimentDim,SerializerDir,TotalOutputDir,2,8)
    # SingleProcess(type,"book",150,sentimentDim,SerializerDir,TotalOutputDir,2,8)
    # SingleProcess(type,"book",150,sentimentDim,SerializerDir,TotalOutputDir,8,2)
    
    #SingleProcess(type,"dvd",150,sentimentDim,SerializerDir,TotalOutputDir,6,4)
    #SingleProcess(type,"music",150,sentimentDim,SerializerDir,TotalOutputDir,9,1)
    #SingleProcess(type,"dvd",150,sentimentDim,SerializerDir,TotalOutputDir,9,1)
    
    #SingleProcess(type,"music",150,sentimentDim,SerializerDir,TotalOutputDir,8,2)
    #SingleProcess(type,"music",150,sentimentDim,SerializerDir,TotalOutputDir,4,6)
    #SingleProcess(type,"dvd",150,sentimentDim,SerializerDir,TotalOutputDir,1,9)
    #SingleProcess(type,"book",150,sentimentDim,SerializerDir,TotalOutputDir,4,6)