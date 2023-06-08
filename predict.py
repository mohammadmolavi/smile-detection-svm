from hog_lbp import hog1 , lbp1
from skimage.transform import resize
import numpy as np


filename = 'finalized_model.sav'


def detect(img , loaded_model):
    hogg = hog1(img)
    lbp2 = lbp1(img)
    img_resized1 = resize(hogg.copy(), (64, 64))
    img_resized2 = resize(lbp2.copy(), (64, 64))
    l=[np.concatenate((img_resized1.flatten() , img_resized2.flatten()) , axis = 0)]
    probability=loaded_model.predict_proba(l)
    return loaded_model.predict(l) , probability





