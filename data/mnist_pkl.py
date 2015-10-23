import cPickle, gzip, theano
import numpy as np
import theano.tensor as T


# prepare the data as Theano shared variables for fast computation
# http://deeplearning.net/tutorial/gettingstarted.html#gettingstarted

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def readData_mnist_pkl():
    """

    :rtype: dictionary of theano.tensor.sharedvar.TensorSharedVariable
    :return: data partitioned into training, validation and test sets. Each with x, y as images (covariates) and
    labels (response).
    """
    # Load the pickled data from disk
    dataFolder = "/Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/Paper_mixedLogistic/software/mixedlogistic/data"
    fileName = "mnist.pkl.gz"
    location = dataFolder + '/' + fileName
    f = gzip.open(location, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # Prepare data
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    return dict(trainx=train_set_x, trainy=train_set_y,
                validx=valid_set_x, validy=valid_set_y,
                testx=test_set_x, testy=test_set_y)


tdata = readData_mnist_pkl()
tdata['testx'].get_value()

# batch_size = 500    # size of the minibatch
#
# # accessing the third minibatch of the training set
# data  = train_set_x[2 * batch_size: 3 * batch_size]
# label = train_set_y[2 * batch_size: 3 * batch_size]