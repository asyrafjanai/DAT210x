import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler

df = pd.read_table('C:/Users/acc_a/Desktop/DAT210x-master/Module6/Datasets/parkinsons.data', sep='\t')
X = df.drop('status', axis=1)
X = df.drop('name', axis=1)
y = df[['status']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7,
                                                    test_size=0.3)
# svc = SVC(gamma=0.005, C=0.1, kernel='linear')
# svc = SVC()
# svc.fit(X_train, y_train.values.ravel())
# score = svc.score(X_test, y_test)
# print(score)


'''
cs = list(np.arange(0.1, 2, 0.1))
gammas = list(np.arange(0.005, 0.1, 0.005))
parameters = {
                'C': cs,
                'gamma': gammas,
                'kernel': ('linear', 'rbf')
             }


reg = GridSearchCV(svc, parameters, verbose=1, scoring='accuracy', cv=4)

reg.fit(X_train, y_train.values.ravel())
print('Best score: %0.3f' % reg.best_score_)
print('Best parameters set:')
best_parameters = reg.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
y_predictions = reg.predict(X_test)
print('accuracy on test set:', accuracy_score(y_test, y_predictions))
'''
scaler = preprocessing.StandardScaler()
# scaler = KernelCenterer()
# scaler = Normalizer()
# scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# pca = PCA(n_components=3)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)


C_range = np.arange(0.05,2.0)
gamma_range = np.arange(0.001,0.1)
best_score = 0
for C in C_range:
    C = C + 0.05
    for gamma in gamma_range:
        gamma = gamma + 0.001

        model = SVC(C=C,gamma=gamma)
        model.fit(X_train,y_train.values.ravel())
        score = model.score(X_test,y_test)

if best_score < score:
   print('Best_Score:', score)
   print('Best_C:',C)
   print('Best_Gamma:', gamma)


# .DOT files can be rendered to .PNGs, if you've already `brew install graphviz`.
# tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)
#
# from subprocess import call
# call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
