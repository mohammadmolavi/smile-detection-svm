import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import cv2
from hog_lbp import hog1 , lbp1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle



Categories = ['smile', 'non_smile']
flat_data_arr = []
target_arr = []
datadir = 'GENKI-R2009a/'

for i in Categories:
    j = 0
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        print(j)
        img_array = imread(os.path.join(path, img))
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        img_array = hog1(gray)
        img_resized = resize(img_array.copy(), (64, 64))
        flat_data_arr.append(img_resized.flatten())
        img_array = lbp1(gray)
        img_resized = resize(img_array.copy(), (64, 64))
        flat_data_arr[-1] = np.concatenate((flat_data_arr[-1] , img_resized.flatten()) , axis=0)
        target_arr.append(Categories.index(i))
        j+=1
    print(f'loaded category:{i} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)


df=pd.DataFrame(flat_data)
df['Target']=target


x=df.iloc[:,:-1]

y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15, random_state=42, stratify=y)


svc = svm.SVC(probability=True)


print("start pipeline")
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('SVC' , svc),
])

pipe.fit(x_train, y_train)

print('Training set score: ' + str(pipe.score(x_train,y_train)))
print('Test set score: ' + str(pipe.score(x_test,y_test)))

filename = 'finalized_model.sav'
pickle.dump(pipe, open(filename, 'wb'))


