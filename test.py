import glob
import numpy as np

car_features = np.load("car_features.p")[:10000]
print("car_file Loaded", len(car_features))

notcar_features = np.load("notcar_features.p")[:10000]
print("notcar_file Loaded ", len(notcar_features))



#
# car_features = np.asarray(notcar_features, dtype=np.float64)
# notcar_features = np.asarray(notcar_features, dtype=np.float64)




print('X.shape', X.shape)

from sklearn.preprocessing import StandardScaler

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

print("now scaled_X step")
scaled_X.dump("scaled_X.p")

