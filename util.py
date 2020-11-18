import numpy as np

""" Rounds all the values of a bounding box (vector with four elements) and outputs as a list """
def box_intr(box):
    return [intr(x) for x in box]

""" Folder should be a pathlib.Path object, then you can do multiple globs and
    get a list as output 
"""
def multi_glob(folder, searches):
    out = []
    for search in searches:
        out.extend(folder.glob(search))
    return out

""" Converts a number to a string with 6 digits """
def long_str(n, l=6):
    # Convert integer n to a string which is always l digits long
    # Assuming the number is not too big of course
    s = str(n)
    extras = l - len(s)
    return '0'*extras + s

# Clips x to be within a <= x <= b
def clip(x, a, b):
    l = [a, x, b]
    l.sort()
    return l[1]

def intr(x):
    return int(round(x))

# From https://stackoverflow.com/questions/48987774/how-to-crop-a-numpy-2d-array-to-non-zero-values    
def smallest_box(a):
    r = a.any(1)
    if r.any():
        m,n = a.shape
        c = a.any(0)
        y1 = r.argmax()
        y2 = m-r[::-1].argmax()
        x1 = c.argmax()
        x2 = n-c[::-1].argmax()
        out = np.array([x1, y1, x2, y2])
    else:
        out = None
    return out

def vector_similarity2(v1, v2):
    delta = v1 - v2
    return 1 / (1 + np.linalg.norm(delta))

# Just a simple dot product of normalized vectors. Range -1 to 1, where 1 means identical.
def vector_similarity(v1, v2):
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    
    if l1 > 0.000001:
        v1 /= l1
    
    if l2 > 0.00001:
        v2 /= l2
    
    return np.dot(v1, v2)
    
# Shows an image on screen until a button is pressed
def show(image, title='Some image or whatever', rgb=False):
    import cv2
    
    if image.dtype == bool:
        image = (image.astype(np.uint8))*255
    
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imshow(title, image)
    cv2.waitKey()
