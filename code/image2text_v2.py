#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import numpy as np
import pandas as pd
from pytesseract import Output
from PIL import Image


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os
import re
import sys
import argparse


# In[3]:

parser = argparse.ArgumentParser(description='This script creates a CSV file that contains the list of elements in the map')
parser.add_argument("filename", help="The filename of the image you use. Example: GrayVersion1.png")

try:
    args = parser.parse_args()
except:
    sys.exit()

if re.search(r'[\w]{1,}[.png]', args.filename):
    filename = re.search(r'([\w]{1,})', args.filename)[0]
else:
    print('Error: Invalid name format')
    sys.exit()

regexp1 = r'^[/"-]{0,}[A-Z0-9]{1,}'
regexp2 = r'[^A-Z]{1}$'
regexp3 = r'^[a-zA-z]{1,}$'

regexp = r'^[0-9a-zA-Z]{,2}.{,1}$'

path = os.getcwd()
image_path = os.path.join(path + '/../dataset/' + filename + '.png')
pytesseract.pytesseract.tesseract_cmd = os.path.join(path + '/../tesseract/tesseract.exe')

if not os.path.isfile(image_path):
    print(f'Error: The file %s.csv does not exist' % (filename))
    sys.exit()


# In[4]:


def merge_data_h(filtered_data):
    ymax = 10
    xmax = 200
    nrow = ['', '', '', '', '', '']
    delete_row = None

    for i in range(len(filtered_data)):
        flag = False
        for j in range(i+1, len(filtered_data)):
            d_y = int(filtered_data[i][2]) - int(filtered_data[j][2])
            d_x = int(filtered_data[i][1]) - int(filtered_data[j][1])
            if abs(d_y) < ymax and abs(d_x) < xmax:
                flag = True
                delete_row = [i, j]
                if d_x < 0:
                    nrow[0] = " ".join([filtered_data[i][0], filtered_data[j][0]])
                else:
                    nrow[0] = " ".join([filtered_data[j][0], filtered_data[i][0]])
                nrow[1] = str(max(int(filtered_data[i][1]), int(filtered_data[j][1])))
                nrow[2] = str(min(int(filtered_data[i][2]), int(filtered_data[j][2])))
                nrow[3] = str(min(int(filtered_data[i][3]), int(filtered_data[j][3])))
                nrow[4] = str(int(filtered_data[i][4]) + int(filtered_data[j][4]))
                nrow[5] = str(int(filtered_data[i][5]) + int(filtered_data[j][5]))
                break

        if flag:
            break
            
    if flag:
        nrow = np.array(nrow).reshape((1,6))
        filtered_data = np.delete(filtered_data, delete_row, axis=0)
        filtered_data = np.concatenate((filtered_data, nrow), axis=0)
    
    return filtered_data, flag


# In[5]:


def merge_data_v(filtered_data):
    ymax = 200
    xmax = 10
    nrow = ['', '', '', '', '', '']
    delete_row = None

    for i in range(len(filtered_data)):
        flag = False
        for j in range(i+1, len(filtered_data)):
            d_y = int(filtered_data[i][2]) - int(filtered_data[j][2])
            d_x = int(filtered_data[i][1]) - int(filtered_data[j][1])
            if abs(d_y) < ymax and abs(d_x) < xmax:
                flag = True
                delete_row = [i, j]
                if d_y < 0:
                    nrow[0] = " ".join([filtered_data[j][0], filtered_data[i][0]])
                else:
                    nrow[0] = " ".join([filtered_data[i][0], filtered_data[j][0]])
                nrow[1] = str(min(int(filtered_data[i][1]), int(filtered_data[j][1])))
                nrow[2] = str(max(int(filtered_data[i][2]), int(filtered_data[j][2])))
                nrow[3] = str(min(int(filtered_data[i][3]), int(filtered_data[j][3])))
                nrow[4] = str(int(filtered_data[i][4]) + int(filtered_data[j][4]))
                nrow[5] = str(int(filtered_data[i][5]) + int(filtered_data[j][5]))
                break

        if flag:
            break
            
    if flag:
        nrow = np.array(nrow).reshape((1,6))
        filtered_data = np.delete(filtered_data, delete_row, axis=0)
        filtered_data = np.concatenate((filtered_data, nrow), axis=0)
    
    return filtered_data, flag


# In[6]:


def merge_data_c(filtered_data):
    th = 300
    regexp1 = r'HA'
    regexp2 = r'^[0-9]{0,}[\/]{0,}[0-9]{0,}.$'

    nrow = ['', '', '', '', '', '']
    delete_row = None

    for i in range(len(filtered_data)):
        flag = False
        for j in range(i+1, len(filtered_data)):
            dd = round(((int(filtered_data[i][1]) - int(filtered_data[j][1])) ** 2 +
                       (int(filtered_data[i][2]) - int(filtered_data[j][2])) ** 2) ** 0.5, 2)
            if (dd < th #and (re.search(regexp1, filtered_data[i][0]) or re.search(regexp1, filtered_data[j][0]))
                #and not (re.search(regexp1, filtered_data[i][0]) and re.search(regexp1, filtered_data[j][0]))
                and ((re.search(regexp2, filtered_data[i][0]) and re.search(regexp1, filtered_data[j][0])) or
                     (re.search(regexp2, filtered_data[j][0]) and re.search(regexp1, filtered_data[i][0])))):
                flag = True
                delete_row = [i, j]
                nrow[0] = " ".join([filtered_data[i][0], filtered_data[j][0]])
                nrow[1] = str(min(int(filtered_data[i][1]), int(filtered_data[j][1])))
                nrow[2] = str(max(int(filtered_data[i][2]), int(filtered_data[j][2])))
                nrow[3] = str(min(int(filtered_data[i][3]), int(filtered_data[j][3])))
                nrow[4] = str(int(filtered_data[i][4]) + int(filtered_data[j][4]))
                nrow[5] = str(int(filtered_data[i][5]) + int(filtered_data[j][5]))
        if flag:
            break

    if flag:
        nrow = np.array(nrow).reshape((1,6))
        filtered_data = np.delete(filtered_data, delete_row, axis=0)
        filtered_data = np.concatenate((filtered_data, nrow), axis=0)
        
    return filtered_data, flag


# In[7]:


def rename_elements(regexp, name, filtered_data, tag_data):
    i = 0
    for row in filtered_data:
        if re.search(regexp, row[0]):
            tag_data[i] = name
        i += 1
    
    return tag_data


# # STEP 1

# In[8]:


originalImage = cv2.imread(image_path)
image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)


# In[9]:


image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]


# In[10]:


image2 = image1.copy()
thresh = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


# In[11]:


# Remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image2, [c], -1, (255,255,255), 5)
    
# Remove vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image2, [c], -1, (255,255,255), 5)


# In[12]:


image3 = image2[200:4200, 240:5700]


# In[13]:


image4 = image3.copy()
thresh = cv2.threshold(image4, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


# In[14]:


# Remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image4, [c], -1, (255,255,255), 5)

# Remove vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image4, [c], -1, (255,255,255), 5)


# In[15]:


#cv2.imwrite(path + '/image4.png', image4)


# In[16]:


d = pytesseract.image_to_data(image4, output_type=Output.DICT)


# In[17]:


n = len(d['text'])
text = np.array(d['text']).reshape((n, 1))
left = np.array(d['left']).reshape((n, 1))
top = np.array(d['top']).reshape((n, 1))
conf = np.array(d['conf']).reshape((n, 1))
width = np.array(d['width']).reshape((n, 1))
height = np.array(d['height']).reshape((n, 1))
data = np.concatenate((text, left, top, conf, width, height), axis=1)


# In[18]:


colorimage = cv2.cvtColor(image4, cv2.COLOR_GRAY2RGB)


# In[19]:


image5 = None

for row in data:
    if int(row[3]) > 0:
        (x, y, w, h) = (int(row[1]), int(row[2]), int(row[4]), int(row[5]))
        image5 = cv2.rectangle(colorimage, (x, y), (x + w, y + h), (0, 255, 0), 2)


# In[20]:


#cv2.imwrite(path + '/image5.png', image5)


# In[21]:


filtered_data = None

for row in data:
    if int(row[3]) > 0:
        if re.search(regexp1, row[0]) and re.search(regexp2, row[0]) and not re.search(regexp3, row[0]):
            if filtered_data is None:
                filtered_data = row
            else:
                filtered_data = np.concatenate((filtered_data, row), axis=0)
                
n_matches = int(filtered_data.shape[0]/6)
filtered_data = filtered_data.reshape((n_matches, 6))


# In[22]:


filtered_data2 = filtered_data.copy()

while (True):
    (filtered_data2, flag) = merge_data_h(filtered_data2)
    if not flag:
        break 


# In[23]:


filtered_data3 = filtered_data2.copy()

while (True):
    (filtered_data3, flag) = merge_data_v(filtered_data3)
    if not flag:
        break 


# In[24]:


filtered_data4 = None

for row in filtered_data3:
    if int(row[3]) > 0:
        if not re.search(regexp, row[0]):
            if filtered_data4 is None:
                filtered_data4 = row
            else:
                filtered_data4 = np.concatenate((filtered_data4, row), axis=0)

n_matches = int(filtered_data4.shape[0]/6)
filtered_data4 = filtered_data4.reshape((n_matches, 6))


# In[25]:


filtered_data5 = filtered_data4.copy()

while (True):
    (filtered_data5, flag) = merge_data_c(filtered_data5)
    if not flag:
        break 


# In[26]:


filtered_data6 = filtered_data5.copy()
filtered_data6 = np.delete(filtered_data6, [3], axis=1)


# In[27]:


f_xmax = (filtered_data6[:,1].astype(int) + filtered_data6[:,3].astype(int)).reshape(len(filtered_data5),1)
filtered_data6 = np.concatenate((filtered_data6, f_xmax), axis=1)


# In[28]:


f_ymax = (filtered_data6[:,2].astype(int) + filtered_data6[:,4].astype(int)).reshape(len(filtered_data5),1)
filtered_data6 = np.concatenate((filtered_data6, f_ymax), axis=1)
filtered_data6 = np.delete(filtered_data6, [3, 4], axis=1) # xmin, ymin, xmax, ymax


# # TAG

# In[29]:


tag_data = ["Unknown" for j in range(len(filtered_data6))]


# In[30]:


tag_regexp = [r'HA', r'^C.*$', r'^P.*$', r'^E.*$', r'^TK.*$', r'(CS|BD)']
tag_name = ['Valve', 'Centrifugal Compressor', 'Pump', 'Exchanger', 'Tank', 'Maine line']

for i in range(6):
    tag_data = rename_elements(tag_regexp[i], tag_name[i], filtered_data6, tag_data)


# In[31]:


filtered_data7 = np.concatenate((filtered_data6, np.array(tag_data).reshape(len(filtered_data6),1)), axis=1)
df = pd.DataFrame(data=filtered_data7, columns=["Tag/ID", "xmin", "ymin", "xmax", "ymax", "Type"])
df[['Tag/ID','Type','xmin','xmax','ymin','ymax']].to_csv(path + '/../output/' + filename + '.csv', index=False,
                                                          sep=',', encoding='utf-8-sig')


# In[ ]:





# In[ ]:





# In[ ]:




