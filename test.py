
import glob
import numpy as np 


###

# car_features =np.asarray( np.load("car_features.p") )
# print("car_file Loaded" , car_features.shape )

# notcar_features = np.asarray( np.load("notcar_features.p"))
# print("notcar_file Loaded ", notcar_features.shape )

# car_features.dump("car_features_RGB_nparr.p")

# notcar_features.dump("notcar_features_RGB_nparr.p")

### 

x = np.array([1,2,3,4,5,8,6])
y = np.array([7,8,9,10,11])


t = np.vstack((x,y))

print(t)