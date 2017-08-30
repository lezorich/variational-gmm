
# coding: utf-8

# In[ ]:




# In[1]:

MACHO_PATH = 'data/macho/'
N_FEATURES = 3


# In[2]:

# mean, color, period
be_lc = np.genfromtxt(MACHO_PATH + 'Be_lc.csv', delimiter=',')
ceph_lc = np.genfromtxt(MACHO_PATH + 'CEPH.csv', delimiter=',')
eb_lc = np.genfromtxt(MACHO_PATH + 'EB.csv', delimiter=',')
longperiod_lc = np.genfromtxt(MACHO_PATH + 'longperiod_lc.csv', 
                              delimiter=',')
microlensing_lc = np.genfromtxt(MACHO_PATH + 'microlensing_lc.csv', 
                                delimiter=',')
quasar_lc = np.genfromtxt(MACHO_PATH + 'quasar_lc.csv', delimiter=',')
rrl_lc = np.genfromtxt(MACHO_PATH + 'RRL.csv', delimiter=',')

non_variables_1 = np.genfromtxt(MACHO_PATH + 'non_variables_1.csv', 
                                delimiter=',')
non_variables_2 = np.genfromtxt(MACHO_PATH + 'non_variables_2.csv', 
                                delimiter=',')
non_variables_3 = np.genfromtxt(MACHO_PATH + 'non_variables_3.csv', 
                                delimiter=',')

# join all non variables lc features into one big matrix
len_1 = non_variables_1.shape[0]
len_2 = non_variables_2.shape[0]
len_3 = non_variables_3.shape[0]

#non_variables_lc = np.zeros((len_1 + len_2 + len_3, N_FEATURES))
#non_variables_lc[0:len_1, :] = non_variables_1
#non_variables_lc[len_1:len_1+len_2, :] = non_variables_2
#non_variables_lc[len_1+len_2:, :] = non_variables_3

def join_lc_features(*args):
    total = 0
    for i in range(len(args)):
        total += args[i].shape[0]
    C = np.zeros(total)
    X = np.zeros((total, N_FEATURES))
    last_idx = 0
    for i in range(len(args)):
        act_idx = last_idx + args[i].shape[0]
        X[last_idx:act_idx, :] = args[i][:, 0:3]
        C[last_idx:act_idx] = i
        last_idx = act_idx
    return X, C

# Now X contain the features, and C contains the class labels, where each class label is the position in the arg
# list when join_lc_features was called.
#X, C = join_lc_features(be_lc, ceph_lc, eb_lc, longperiod_lc, microlensing_lc, quasar_lc, rrl_lc, non_variables_lc)
X, C = join_lc_features(be_lc, ceph_lc, eb_lc, longperiod_lc, microlensing_lc, quasar_lc, rrl_lc)

# normalize the data
X = (X - X.mean(axis=0)) / X.std(0)


# ## Training and testing data
# 
# In the cell below, we separate our matrix $X$ into training and testing datasets. We choose randomly $10\%$ of the total number of lightcurves as our testing dataset.

# In[3]:

TRAINING_FRACTION = 0.9

N = C.shape[0]
N_TRAINING = int(TRAINING_FRACTION * N)
N_TESTING = N - N_TRAINING

shuffle_idxs = np.arange(N)
np.random.shuffle(shuffle_idxs)
idx_training = shuffle_idxs[:N_TRAINING]
idx_testing = shuffle_idxs[N_TRAINING:]

X_training = X[idx_training, :]
X_testing = X[idx_testing, :]
C_training = C[idx_training]
C_testing = C[idx_testing]


# In[4]:

print("Number training:", N_TRAINING)
print("Number testing:", N_TESTING)


# In[5]:

M = 7
K = 2
N_FEATURES = 3


# In[6]:

from streaming_gmm.streaming_variational_gmm import VariationalGMM

models = [VariationalGMM(K, N_FEATURES) for m in range(M)]

for m in range(M):
    index = np.where(C_training == m)
    models[m].fit(np.squeeze(X_training[index, :]))


# In[7]:

def predict(testing):
    n_testing = testing.shape[0]
    estimated_classes = np.zeros(n_testing).astype(int)
    for i in range(n_testing):
        estimated_i = 0
        max_p = -1e7
        for m in range(M):
            p = models[m].predict(testing[i])
            if p > max_p:
                estimated_i = m
                max_p = p
        estimated_classes[i] = estimated_i
    return estimated_classes

def calculate_accuracy(estimated_class, classes):
    return np.sum(estimated_class == classes.astype(int)) / classes.shape[0]

print(np.min(C_training))
print("Training accuracy:", calculate_accuracy(predict(X_training), C_training))
print("Testing accuracy:", calculate_accuracy(predict(X_testing), C_testing))
#print(X_testing.shape)


# In[ ]:



