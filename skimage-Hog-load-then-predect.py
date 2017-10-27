import numpy as np
import pickle

svcdata  = pickle.load(open("saved_svc.p", 'rb'))
X_test  = pickle.load(open("X_test.p", 'rb'))
y_test  = pickle.load(open("y_test.p", 'rb'))

print(svcdata.predict(X_test[0]))
#
# print(len(X_test))
#
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,svcdata.predict(X_test),normalize=False))
# print(accuracy_score(y_test,svcdata.predict(X_test),normalize=True))