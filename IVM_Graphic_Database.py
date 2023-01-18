# IVM Graphic Database

# Importeren van basis bibliotheken
import streamlit as st
import numpy as np

# Basis instellingen
st.set_page_config(
    page_title="IVM Graphic Database", 
    page_icon="🧊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('IVM Graphic Database')

# Andere bibliotheken importeren
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio
import scipy.misc
import cv2
import os
import scipy
import streamlit as st

# Uploaden van bestanden en deze opslaan in de map
col1, _ = st.columns([2, 1])

with col1:
    def load_image(image_file):
        img = Image.open(image_file)
        return img

    image_file = st.file_uploader("Upload hier jouw grafische werk",
                              type=["jpg","jpeg"])

    if image_file is not None:
        file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
        st.write(file_details)
        st.image(load_image(image_file), width=250)
    
        with open(os.path.join("C:\\Users\\loisg\\OneDrive - Windesheim Office365\\Leerjaar 3\\Minor - Big Data & Design\\Block 3\\dataset\\",image_file.name),"wb") as f:
            f.write((image_file).getbuffer())
        
            origineel = st.checkbox('This assignment submission is my own, original work.')

            waarschuwing = '<p style="font-family:Courier; color:red; font-size: 16px;">You must agree to the submission pledge before you can submit this assignment.</p>'

            if origineel:
                st.write('')
            else:
                st.write(waarschuwing, unsafe_allow_html=True) #Rode kleur geven

            if st.button('Submit assignment'):
                st.write('Submitted succesfully, paste this link into Canvas: www.submission.com/succesfull/164929')
			  
            st.success("File Saved")
              
# Plagiaatcontrole uitvoeren
IMAGE_DIR = 'C:\\Users\\loisg\\OneDrive - Windesheim Office365\\Leerjaar 3\\Minor - Big Data & Design\\Block 3\\dataset\\'

os.chdir(IMAGE_DIR)
os.getcwd()

image_files = os.listdir()

imageio.imread(image_files[0]).shape

def filter_images(images):
    image_list = []
    for image in images:
        try:
            assert imageio.imread(image).shape[2] == 3
            image_list.append(image)
        except  AssertionError as e:
            print(e)
    return image_list

def img_gray(image):
    image = imageio.imread(image)
    return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)

def resize(image, height=30, width=30):
    row_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten('F')
    return row_res, col_res

def intensity_diff(row_res, col_res):
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()
    return difference_row
    return np.vstack((difference_row, difference_col)) #str method

def difference_score(image, height = 30, width = 30):
    gray = img_gray(image)
    row_res, col_res = resize(gray, height, width)
    difference = intensity_diff(row_res, col_res)
    
    return difference
    
def hamming_distance(image, image2):
    score =scipy.spatial.distance.hamming(image, image2)
    return score

def difference_score_dict(image_list):
    ds_dict = {}
    duplicates = []
    for image in image_list:
        ds = difference_score(image)
        
        if image not in ds_dict:
            ds_dict[image] = ds
        else:
            duplicates.append((image, ds_dict[image]) )
    
    return  duplicates, ds_dict

image_files = filter_images(image_files)
duplicates, ds_dict =difference_score_dict(image_files)

len(duplicates)

len(ds_dict.keys())

import itertools
import scipy.spatial
for k1,k2 in itertools.combinations(ds_dict, 2):
    if hamming_distance(ds_dict[k1], ds_dict[k2])< .10:
        duplicates.append((k1,k2))
        
len(duplicates)

for file_names in duplicates[:10]:
    try:
    
        plt.subplot(121),plt.imshow(imageio.imread(file_names[0]))
        plt.title('Duplicaat'), plt.xticks([]), plt.yticks([])

        plt.subplot(122),plt.imshow(imageio.imread(file_names[1]))
        plt.title('Origineel'), plt.xticks([]), plt.yticks([])
        st.pyplot()
    
    except OSError as e:
        continue
    
