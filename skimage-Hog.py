import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
import pickle
# from dataHandling import dataBase 
# Positive, Negative = dataBase()

# from skimage import color, exposure, io
# from skimage.feature import hog

# def extract_feature(images):
#     features = np.array([])
#     for image in images:
#         sourceImage = color.rgb2gray( io.imread(image) )
#         hog_features = hog(sourceImage, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=False, feature_vector=True) 
#         features = np.append(features,hog_features)
#         print(hog_features.ndim)
        
#         print(image)
         
#     return features

# car_features = extract_feature(Positive)
# car_features.dump("car_features_RGB.dat")

# notcar_features = extract_feature(Negative)
# notcar_features.dump("notcar_features_RGB.dat")

###

car_features = np.load("car_features.p")
print("car_file Loaded" , len(car_features ) )

notcar_features =  np.load("notcar_features.p")
print("notcar_file Loaded ", len(notcar_features) )

minNump = 130000 # np.min(np.array([len(car_features), len(notcar_features)]))
print(minNump)
car_features = car_features[:minNump]
print("car_file Loaded" , len(car_features ))
notcar_features = notcar_features[:minNump]
print("notcar_file Loaded ", len(notcar_features) )

### 

length = len(sorted(car_features,key=len, reverse=True)[0])
notlength = len(sorted(notcar_features,key=len, reverse=True)[0])

length = max( length , notlength )

car_features= np.array([np.hstack((xi , [0]*(length-len(xi)))) for xi in car_features])

pickle.dump(car_features, open("car_features_generalisedList.p", 'wb'), protocol=4)

notcar_features= np.array([np.hstack((xi , [0]*(length-len(xi)))) for xi in notcar_features])

pickle.dump(notcar_features, open("notcar_features_generalisedList.p", 'wb'), protocol=4)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Create an array stack of feature vectors
# X = np.concatenate( ( car_features, notcar_features ) )
# print( 'X.shape', X.shape )
# X = np.array([car_features, notcar_features])

# X = np.vstack(list(X[:,:]))
# X = np.asfarray(X)
print( 'X.shape', X.shape )

from sklearn.preprocessing import StandardScaler

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

print("now scaled_X step")
np.asarray(scaled_X).dump("saved_X_scaler.dat")

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)


# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)

pickle.dump(svc, open("saved_svc.p", 'wb'), protocol=4)
pickle.dump(X_test, open("X_test.p", 'wb'), protocol=4)
pickle.dump(y_test , open("y_test.p", 'wb'), protocol=4)


# """
# car_file Loaded 337780
# notcar_file Loaded  358071
# 337780
# car_file Loaded 337780
# notcar_file Loaded  337780
# X.shape (675560, 1760)
# Traceback (most recent call last):
#   File "C:\Program Files\JetBrains\PyCharm Community Edition 2017.2.3\helpers\pydev\pydevd.py", line 1599, in <module>
#     globals = debugger.run(setup['file'], None, None, is_module)
#   File "C:\Program Files\JetBrains\PyCharm Community Edition 2017.2.3\helpers\pydev\pydevd.py", line 1026, in run
#     pydev_imports.execfile(file, globals, locals)  # execute the script
#   File "C:\Program Files\JetBrains\PyCharm Community Edition 2017.2.3\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
#     exec(compile(contents+"\n", file, 'exec'), glob, loc)
#   File "D:/Privet/work/Python/PycharmProjects/parking-car-counter/skimage-Hog.py", line 72, in <module>
#     X_scaler = StandardScaler().fit(X)
#   File "C:\Program Files\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py", line 557, in fit
#     return self.partial_fit(X, y)
#   File "C:\Program Files\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py", line 621, in partial_fit
#     self.n_samples_seen_)
#   File "C:\Program Files\Anaconda3\lib\site-packages\sklearn\utils\extmath.py", line 742, in _incremental_mean_and_var
#     new_unnormalized_variance = X.var(axis=0) * new_sample_count
#   File "C:\Program Files\Anaconda3\lib\site-packages\numpy\core\_methods.py", line 101, in _var
#     x = asanyarray(arr - arrmean)
# MemoryError
#
# """