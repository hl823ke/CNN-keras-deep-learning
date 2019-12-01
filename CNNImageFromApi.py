from PIL import Image
import time
import csv
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.io 
import urllib
import os
import json 
import pyvo as vo
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from skimage.transform import resize
from skimage.transform import rescale
import PIL.Image as Image
import requests
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adam
pd.set_option('display.max_colwidth', -1)
# do *not* show python warnings 
import warnings
warnings.filterwarnings('ignore')


def GetImage(ra,dec,sigma, fold):
    url = 'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=first&'

    try:
#         data_url = vo.imagesearch(url,pos=(ra, dec), size=0.1, format="image/fits")#, timeout=500)
#         img_url = data_url[0].getdataurl()

        img=vo.imagesearch(url,pos=(ra, dec), size=0.1, format="image/fits")#, timeout=500)
        dataurl= img[0].getdataurl()
        
        resp = requests.get(dataurl)#timeout = 120
        img = fits.open(io.BytesIO(resp.content))
    
    except (ValueError, IndexError):
        raise
        pass

    except urllib.error.URLError: # RuntimeError: #URLError:# , NameError
        raise
        print("Time out!, Re-run please.")
    #         raise
        pass

    image = img[0].data
    img.close()
    image = np.squeeze(image)
    img = np.copy(image)
#     image = np.array(image)
    idx = np.isnan(img)#pd.isnull(img)#
    img[idx] = 0
    sigma = sigma#3.0
    
    # Estimate stats
    mean, median, std = sigma_clipped_stats(img, sigma=sigma, iters=10)
    # Clip off n sigma points
    lim = std*sigma
    img[img<lim]= 0.0#clip(img,std*sigma)
    img_clip = img
#     img_clip = img_clip[ 75:225, 75:225]
        
#     img = (((img_clip - img_clip.min()) / (img_clip.max() - img_clip.min())) * 255.9)#.astype(np.uint8)
    minval, maxval = img_clip.min(),img_clip.max()
    norm = img_clip - minval
    img = norm*(1./(maxval-minval))#.astype(np.uint8)
    imsave(fold+str(ra)+'.png',img)
    return #img,dataurl#img

