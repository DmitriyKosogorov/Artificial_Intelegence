import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops
import numpy as np


def extract_features(image):
    features = []
    labeled = label(image)
    region = regionprops(labeled)[0]
    features.append(region.eccentricity)
    filling_factor = region.area / region.bbox_area
    features.append(filling_factor)
    features.append(len(region.filled_image[region.filled_image != 0])/  region.bbox_area)
    centroid=[region.local_centroid[0]/region.image.shape[0],region.local_centroid[1]/region.image.shape[1]]
    features.extend(centroid)
    return features

def image_to_text(image,knn):
    text=""
    mid_dif=[]
    spa_dif=[]
    middle_dif=0
    space_dif=0
    graying=image.copy()
    graying[graying>0]=1
    labeled=label(graying)
    regions=sorted(regionprops(labeled),key=lambda r: r.bbox[1],reverse=False)
    for i in range(len(regions)-1):
        mid_dif.append((regions[i+1].bbox[3]-regions[i].bbox[3]))
        spa_dif.append((regions[i+1].bbox[1]-regions[i].bbox[3]))
    for value in mid_dif:
        middle_dif+=value
    middle_dif=middle_dif/len(mid_dif)/9
    
    for value in spa_dif:
        if(value>0):
            space_dif+=value
    space_dif=space_dif/len(spa_dif)*2
    skip=False
    for i in range(0,len(regions),1):
        if(skip==False):
            region=regions[i]
            y1,x1,y2,x2=region.bbox
            if(i<len(mid_dif) and mid_dif[i]<middle_dif): 
                symbol=image[min(regions[i].bbox[0],regions[i+1].bbox[0]):max(regions[i].bbox[2],regions[i+1].bbox[2]),min(regions[i].bbox[1],regions[i+1].bbox[1]):max(regions[i].bbox[3],regions[i+1].bbox[3])]   
                skip=True
            else:
                symbol=image[y1:y2,x1:x2]
            if(i==20):
                plt.imshow(symbol)
            gray=symbol.copy()
            binary = gray.copy()
            binary[gray > 0] = 1
            
            test_symbol=extract_features(binary)
            test_symbol=np.array(test_symbol,dtype='f4').reshape(1,5)
            ret,result,neighbours,dist=knn.findNearest(test_symbol, 3)
            letter=chr(int(ret))
            text=text+letter
        else:
            skip=False
        if(i<len(spa_dif) and spa_dif[i]>space_dif):
            text=text+' '
    return(text)
        

train_dir=Path('out')/"train"
train_data=defaultdict(list)

for path in sorted(train_dir.glob('*')):
    if path.is_dir():
        for img_path in path.glob('*.png'):
            symbol=path.name[-1]
            image = cv2.imread(str(img_path),0)
            binary=image.copy()
            binary[binary>0]=1
            train_data[symbol].append(binary)
            



features_array=[]
responses=[]
for i,symbol in enumerate(train_data):
    for img in train_data[symbol]:
        features=extract_features(img)
        features_array.append(features)
        responses.append(ord(symbol))
        
features_array=np.array(features_array, dtype='f4')
responses=np.array(responses)


knn=cv2.ml.KNearest_create()
knn.train(features_array, cv2.ml.ROW_SAMPLE, responses)


for i in range(6):
    filename=str(i)+'.png'
    img_path=Path('out')/filename
    image=cv2.imread(str(img_path),0)
    print(image_to_text(image,knn))