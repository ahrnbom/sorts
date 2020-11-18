import numpy as np
import pycocotools.mask as rle
from pathlib import Path

# pmasks should be on format [PEOPLE, HEIGHT, WIDTH] as provided by Detectron2
def encode(pmasks):
    # Convert to COCO's format
    pmasks = np.asfortranarray(np.moveaxis(pmasks, 0, -1))
    
    # Convert to bytes with run-length encoding
    Rs = rle.encode(pmasks.astype(np.uint8))
    
    # Format: first line contains the size, then each line is just ascii
    # The size is just "HEIGHT,WIDTH"
    R_list = [f"{Rs[0]['size'][0]},{Rs[0]['size'][1]}"]
    R_list.extend([R['counts'].decode('ascii') for R in Rs])
    text = '\n'.join(R_list)
    
    return text
    
def encode_one(mask):
    mask = np.asfortranarray(np.expand_dims(mask, -1))
    Rs = rle.encode(mask)
    return Rs[0]['counts'].decode('ascii')

def decode(text):
    lines = text.split('\n')
    lines = [x for x in lines if x] # Remove empty lines
    
    # First line is the size
    size = [int(x) for x in lines[0].split(',')]
    
    # The remaining are bytes encoded as ascii
    byte_lines = [bytes(x, 'ascii') for x in lines[1:]]
    
    bmasks = list()
    for b in byte_lines:
        bdict = dict()
        bdict['size'] = size
        bdict['counts'] = b
        bmasks.append(bdict)
    bmasks = rle.decode(bmasks)
    
    return bmasks

def decode_one(text, size):
    bdict = dict()
    bdict['size'] = size
    bdict['counts'] = text
    
    return rle.decode([bdict])[:, :,0].astype(bool)
    
def write_file(path, pmasks):
    path.write_text(encode(pmasks))

def read_file(path, dtype=np.uint8):
    text = path.read_text()
    if text:
        return decode(text).astype(dtype)
    else:
        return np.empty((0,0,0), dtype=bool)
