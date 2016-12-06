from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import sparse, io
from sklearn.decomposition import PCA


def dimensionality_reduction(training_data, test_data, type='pca'):
    if type == 'pca':
        n_components = 1000
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        pca.fit(training_data)
        print("done in %0.3fs" % (time() - t0))
        t0 = time()
        training_data_transform = sparse.csr_matrix(pca.transform(training_data))
        test_data_transform = sparse.csr_matrix(pca.transform(test_data))
        print("done in %0.3fs" % (time() - t0))
        #random_projections
        #feature_agglomeration
        return training_data_transform, test_data_transform



def split_data(content, label):
    training_data, test_data, training_target, test_target = train_test_split(
        content, label, test_size=0.1, random_state=20)
    return training_data, test_data, training_target, test_target

def standardized_data(content, label):
    training_data, test_data, training_target, test_target = split_data(content, label)
    scalar = preprocessing.StandardScaler().fit(training_data)
    training_data_transformed = scalar.transform(training_data)
    test_data_transformed = scalar.transform(test_data)
    return training_data_transformed, test_data_transformed, training_target, test_target