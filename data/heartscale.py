import numpy as np
import scipy.sparse as sparse

__author__ = 'panc'


"""
Read Cj Lin's heart scale data as a sparse matrix.
"""
# Read the data file as list of strings
folder = "/Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/Paper_mixedLogistic/software/mixedlogistic/data"
fileName = "heart_scale"
location = folder + "/" + fileName

def readData_heartScale(location):
    with open(location) as f:
        content = f.readlines()

    f.close()

    stripedContent = [x.strip().split(' ') for x in content]
    # temp = [x.strip().split(' ') for x in content][0]

    def splitSample(sampleAsList):
        label = int(sampleAsList[0])
        covariates = sampleAsList[1:]
        keys = []
        values = []
        for i in range(len(covariates)):
            keyValuePair = covariates[i].split(':')
            keys.append(int(keyValuePair[0]))
            values.append(float(keyValuePair[1]))

        return label, keys, values

    # splitSample(temp)

    splitedContent = [splitSample(sample) for sample in stripedContent]
    labels = []
    samples = []
    for i in range(len(splitedContent)):
        labels.append(splitedContent[i][0])
        samples.append(splitedContent[i][1:])
    labels = (np.array(labels) > 0).astype(int)

    # labels
    # len(samples[0][1])
    _cols = [x[0] for x in samples]
    cols = np.array([item for sublist in _cols for item in sublist]) - 1
    d = [len(elem) for elem in _cols]
    # len(d)
    rows = np.repeat(range(len(samples)), d)
    _values =[x[1] for x in samples]
    values = np.array([item for sublist in _values for item in sublist])

    covariatesAsSparseMatirx = sparse.coo_matrix((values, (rows, cols)), shape=(len(samples), 13))

    # covariatesAsSparseMatirx.toarray()[0,:]
    # samples[0]
    return dict(labels=labels, covariates=covariatesAsSparseMatirx)

# data = readData_heartScale(location)
# data['labels']
# data['covariates']