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
from scipy.misc import imsave
pd.set_option('display.max_colwidth', -1)
# do *not* show python warnings 
import warnings
warnings.filterwarnings('ignore')
from astropy.coordinates import Angle
from time import sleep




def GetImage(ra,dec,sigma, fold):
    url = 'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=first&'

    try:
        img=vo.imagesearch(url,pos=(ra, dec), size=0.1, format="image/fits")#, timeout=500)
        print(img)
        dataurl= img[0].getdataurl()
        
        resp = requests.get(dataurl)#timeout = 120
        print(resp.content)
        img = fits.open(io.BytesIO(resp.content), ignore_missing_end=True)
    
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
    sigma = 3#3.0
    
    # Estimate stats
    mean, median, std = sigma_clipped_stats(img, sigma=sigma, iters=10)
    # Clip off n sfigma points
    lim = std*sigma
    img[img<lim]= 0.0#clip(img,std*sigma)
    img_clip = img
        
    minval, maxval = img_clip.min(),img_clip.max()
    norm = img_clip - minval
    img = norm*(1./(maxval-minval))#.astype(np.uint8)
    imsave('/Users/haikristianlethanh/Desktop/Fetched/Fri'+str(ra)+'.png',img)
    return True 

gals_radio = pd.read_csv("/Users/haikristianlethanh/Downloads/FRI-cat (1).csv",sep=',')
gals_radio
df=pd.DataFrame(gals_radio)
df.head()



start = time.time()
gals_radio[6:9]

t=0
 =============================================================================
 for i, cord in gals_radio[:6].iterrows():
     t+=1     
     RA = cord[1]
     DEC = cord[2]
     print(t,": ",RA, DEC)
     GetImage(RA, DEC, 3, "FRI/")
     time.sleep(1.5)
#     print(str(cord[0]) + 'h' + str(cord[1]) + 'm' + str(cord[2]) + 's')
#     print(str(cord[3]) + '°' + str(cord[4]) + '′' + str(cord[5]) + '″')
#     suradnice_RA = str(cord[0]) + 'h' + str(cord[1]) + 'm' + str(cord[2]) + 's'
#     suradnice_DEC = str(cord[3]) + '°' + str(cord[4]) + '′' + str(cord[5]) + '″'
#     ra= Angle(suradnice_RA);
#     dec= Angle(suradnice_DEC);
 =============================================================================

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

fits_image_filename = fits.util.get_testdata_filepath('/Users/haikristianlethanh/Downloads/first_14dec17.fits')


import multiprocessing as mp
file = "/Users/haikristianlethanh/Downloads/first_14ec17_upravene.csv"
gals_radio = pd.read_csv(file)

start = time.time()

pool = mp.Pool(mp.cpu_count())

[pool.apply(GetImage, args=(cord[1], cord[2], 3, "FRI/")) for i, cord in gals_radio[:6].iterrows()]

pool.close()

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))