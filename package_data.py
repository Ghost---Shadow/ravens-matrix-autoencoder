'''
Reads all data from the sample folders and packages them into a npy file
'''

import numpy as np
import cv2

PATH = './data/'
FOLDERS = [
    'sample0/',
    'sample1/',
    'sample2/',
    'sample3/',
    'sample4/',
    'sample5/',
    'sample6/',
    'sample7/',
    'sample8/',
    'sample9/',
    'sample10/',
    'sample11/',
]
FILES = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', \
         'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']

IMG_SIZE = 64
NUM_SAMPLES = len(FOLDERS)
NUM_FILES = len(FILES)

DATA = np.zeros([NUM_SAMPLES, NUM_FILES, IMG_SIZE, IMG_SIZE])

for i, folder in enumerate(FOLDERS):
    for j, file in enumerate(FILES):
        full_path = PATH + folder + file + '.png'
        DATA[i, j] = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        print(full_path)

np.save(PATH + 'processed.npy', DATA)
