import glob 

def dataBase():
    imageEXT = '*.[jpg,png,gif,bmp]*'
    DataSet_Path = "/media/maged/B01CA94B1CA90E02/Users/maged.rawash/Downloads/car/PKLot/PKLot/PKLotSegmented/"

    pathList_Negative = glob.iglob(DataSet_Path +"**/Empty/"+imageEXT, recursive=True)
    pathList_Positive = glob.iglob(DataSet_Path +"**/Occupied/"+imageEXT, recursive=True)

    # for img in pathList_Positive:
    #     print(img)

    
    return pathList_Positive , pathList_Negative

