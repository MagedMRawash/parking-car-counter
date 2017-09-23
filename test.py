
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

x = np.array([1,2,3,4,5,6,55,8,33,34,85])
y = np.array([7,8,9,10,11])


t1 = np.stack((x,y))
t2 = np.asarray((x,y))

X = np.concatenate( (x,y) )
    
print('X.shape',X.shape)
print(t1)
print(t2)
print(X)