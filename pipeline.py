# IMPORTING AND CLEANING DATA
import pandas as pd
import numpy as np
df = pd.read_csv('./main.csv')
df['Time'] = pd.to_datetime(df['Time'])
dft = df.copy()
dft.drop(dft[dft.isnull().sum(axis = 1) > 3].index, axis = 0, inplace = True)
dft.drop(dft[dft['HG_ARRAY_02'].isnull()].index, axis = 0, inplace = True)
dft.drop(dft[dft['HG_ARRAY_02'] < 500].index, axis = 0, inplace = True)
dft.drop(dft[dft['Wind_Velocity'] > 10].index, axis = 0, inplace = True)
for index, row in dft[dft['Wind_Velocity'].isnull()].iterrows():
    dft.loc[index, 'Wind_Velocity'] = np.random.normal(dft['Wind_Velocity'].mean(), dft['Wind_Velocity'].std())
for index, row in dft[dft['TT_16'].isnull()].iterrows():
    dft.loc[index, 'TT_16'] = dft.loc[index, 'TT_11'] + np.random.normal((dft['TT_16'] - dft['TT_11']).mean(), (dft['TT_16'] - dft['TT_11']).std())
for index, row in dft[dft['PYNM_02'].isnull()].iterrows():
    dft.loc[index, 'PYNM_02'] = np.random.normal(dft['PYNM_02'].mean(), dft['PYNM_02'].std())

# PIPELINE
import pickle
import time
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
ds = dft.values
traindata, testdata, trainanswers, testanswers = train_test_split(
    dft.iloc[:, 2:7], dft.iloc[:, 7], test_size = 0.4)
while True: # Each iteration is one model training instance. A condition necessary here to stop the code when wanted, maybe we could scan the time for which this loop runs.
    n1, n2 = np.random.randint(3, 21), np.random.randint(3, 21)
    lr = 'constant' if np.random.randint(0, 2) == 0 else 'adaptive'
    maxiter, iternochange = np.random.randint(50000, 150000), np.random.randint(5000, 25000)
    model = MLPRegressor(hidden_layer_sizes = (n1, n2,),
                         learning_rate = lr,
                         max_iter = maxiter,
                         verbose = False,
                         early_stopping = True,
                         validation_fraction = 0.2,
                         n_iter_no_change = iternochange)
    st = time.time()
    model.fit(traindata, trainanswers)
    et = time.time()
    trainpredictions = model.predict(traindata)
    trainr2 = r2_score(trainanswers, trainpredictions)
    testpredictions = model.predict(testdata)
    testr2 = r2_score(testanswers, testpredictions)
    metrics = pd.read_csv('metrics.csv')
    if testr2 > metrics['testr2'].max(): # If this condition is passed, the 'bestmodel.nnm' file is updated with current model. A better condition necessary here which also takes into account the size of data set.
        pickle.dump(model, open('bestmodel.nnm', 'wb'))
    metrics = metrics.append(pd.DataFrame([[n1, n2, lr, trainr2, testr2, et - st, model.n_iter_, maxiter, iternochange, dft.shape[0]]],
                                          columns = ['HL1', 'HL2', 'learning', 'trainr2', 'testr2', 'time', 'iterations', 'max_iter', 'n_iter_no_change', 'datasetsize']),
                             sort = False)
    metrics.to_csv(r'metrics.csv', index = False)
    print(metrics.shape, end = ' | ')

