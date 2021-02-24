import numpy as np
import cv2
from pathlib import Path
import imageio as iio
import sys
import multiprocessing
from numba import jit

import full_format
from colors import get_colors
from util import long_str, smallest_box, intr

""" Draw a bounding box """    
def draw_box(im, color, x1, y1, x2, y2, text=None):
    im = cv2.rectangle(im, (x1,y1), (x2,y2), color, 3)
    if text is not None:
        im = cv2.putText(im, text, (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return im

""" Apply a mask with 50% transparency (optimized)
    Yes, this is faster than np.copyto
"""
@jit(nopython=True)
def apply_mask_50(im, mask, color):
    ys, xs = np.where(mask)
    for y,x in zip(ys, xs):
        im[y,x,:] = im[y,x,:]//2 + color//2
    return im

""" Applies a mask without transparency """
def apply_mask_opaque(im, mask, color):
    for i in range(3):        
        np.copyto(im[:,:,i], color[i] * np.ones_like(im[:,:,i]), where=mask)
    return im

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

""" Draw a mask on an image with 50% transparency and a border and centered text """
def draw_mask(im, mask, color, text, is_other=False, extra_text_scale=1.0):
    np_color = np.array(color, dtype=np.uint8)
    im = apply_mask_50(im, mask, np_color)
    k = kernel
    mask_border = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, k).astype(bool)
    im = apply_mask_opaque(im, mask_border, np_color)
    
    box = smallest_box(mask)
    if box is None:
        return im
    
    dy = 0
    if is_other:
        dy = 15
    
    x = intr((box[0] + box[2])/2)
    y = intr((box[1] + box[3])/2 + dy)
    
    font = cv2.FONT_HERSHEY_PLAIN
    text_scale = 0.75*im.shape[0]/480.0
    if text_scale < 1.0:
        text_scale = 1.0
    text_scale *= extra_text_scale
    text_w = cv2.getTextSize(text, font, text_scale, 1)[0][0]
    line = cv2.LINE_AA
    cv2.putText(im, text, (x - text_w//2,y), font, text_scale, (0,0,0), 3, line)
    cv2.putText(im, text, (x - text_w//2,y), font, text_scale, color,   1, line)
    
    return im

""" Visualize direct output from Detectron2 """
def visualize_direct(im, persons):
    n = persons.shape[0]
    h = persons.shape[1]
    w = persons.shape[2]
    colors = get_colors(n)
    
    masks = np.zeros((h,w,3), dtype=np.uint8)
    
    for i in range(n):
        person = persons[i, ::]
        color = colors[i]
        for j in range(3):
            color_channel = np.uint8(color[j])
            colored_person = color_channel*person
            masks[:,:,j] += colored_person
            
    return cv2.addWeighted(im, 0.5, masks, 0.5, 0.0)
    
""" Visualize people data saved to .txt file """
def visualize_saved_people(im_path, text_file, out_path):
    im = iio.imread(im_path)
    h = im.shape[0]
    w = im.shape[1]
    
    text = text_file.read_text()
    bmasks = full_format.decode(text)        
    n = bmasks.shape[-1]    
    colors = get_colors(n)
    masks = np.zeros((h,w,3), dtype=np.uint8)
            
    for i in range(n):
        color = colors[i]
        for j in range(3):
            color_channel = np.uint8(color[j])
            bmask = bmasks[:, :, i]
            
            # Optional: compute contour
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            bmask = cv2.morphologyEx(bmask.copy(), cv2.MORPH_GRADIENT, kernel)
            
            masks[:, :, j] += color_channel*bmask
    
    frame = cv2.addWeighted(im, 0.5, masks, 0.5, 0.0)
    frame = cv2.putText(frame, im_path.stem, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    iio.imwrite(out_path, frame)

def visualize_tracks_images_compare(t1_path, t2_path, out_path, im_folder):
    def get_image(i):
        return iio.imread(im_folder / f"{long_str(i)}.jpg")
    visualize_tracks(t1_path, out_path, get_image, n_colors=32, other=t2_path)
    

def visualize_tracks_images(tracks_path, out_path, im_folder):
    def get_image(i):
        return iio.imread(im_folder / f"{long_str(i)}.jpg")
    visualize_tracks(tracks_path, out_path, get_image, n_colors=32)
        
def visualize_tracks_video(tracks_path, out_path, video_file):
    with iio.get_reader(video_file) as invid:
        def get_image(i):
            return invid.get_data(i)
        visualize_tracks(tracks_path, out_path, get_image, n_colors=16)
    
def str_val(x):
    if x == 'main':
        return 0
    if x == 'other':
        return 0.5

def visualize_tracks(tracks_path, out_path, get_image, should_print=True, n_colors=16, other=None):
    colors = get_colors(n_colors)

    tracks = tracks_path.read_text().split('\n')
    lines = [(line,'main') for line in tracks if line]
    
    if other is not None:
        other_tracks = other.read_text().split('\n')
        lines.extend([(line,'other') for line in other_tracks if line])
        lines.sort(key=lambda x: int(x[0].split(' ')[0]) + str_val(x[1]))
        
    curr_im = None
    curr_frame = -1
        
    with iio.get_writer(out_path, fps=10) as outvid:
        for line,origin in lines:
                
            frame_no, track_id, _, h, w, mask_str = line.split(' ')
            
            frame_no = int(frame_no)
            track_id = int(track_id)
            h = int(h)
            w = int(w)
            
            if not curr_frame == frame_no:
                # If old frame exist, write it first!
                if curr_im is not None:
                    outvid.append_data(curr_im)
                    
                # Load new image from video/file
                curr_im = get_image(frame_no)
                curr_frame = frame_no
                
                if should_print:
                    print('visualizing', out_path, curr_frame)
                    
            mask = full_format.decode_one(mask_str, (h,w))
            if track_id == 10000:
                color = (200,200,200)        
                text = 'IGNORE'
            else:
                color = colors[track_id % n_colors]
                text = str(track_id)[1:]
            
            is_other = False
            if origin == 'other':
                color = (color[0]//2, color[1]//2, color[2]//2)
                is_other = True
            
            curr_im = draw_mask(curr_im, mask, color, text, is_other=is_other)

        outvid.append_data(curr_im) # Final frame too!



def main():
    if len(sys.argv) < 2:
        print("Usage: `python visualize.py RUN_NAME`")
        print("RUN_NAME can also be a folder containing text files of detections")
        print("also, two run names can be provided for a comparison")
        sys.exit(1)
        
    run_name = sys.argv[-1]
    
    if (run_name == 'gt'):
        folder = Path('gt')
        if not folder.is_dir():
            raise FileNotFoundError(folder) 
    elif Path(run_name).is_dir():
        for text_file in Path(run_name).glob('*/*/*.txt'):
            im_file = Path('MOTS') / 'train' / text_file.parent.parent.name / 'img1' / text_file.with_suffix('.jpg').name
            out_file = text_file.with_suffix('.png')
            visualize_saved_people(im_file, text_file, out_file)
            print(out_file)
    else:
        folder = Path('output') / run_name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        
    vis_folder = folder / 'vis'
    if not vis_folder.is_dir():
        vis_folder.mkdir()
    
    # Check if run uses train or test set
    the_set = None
    if (run_name == 'gt'):
        the_set = 'train'
    else:
        for line in (folder / 'this_run.settings').read_text().split('\n'):
            if line.startswith('set: '):
                the_set = line.split(' ')[-1]
                break
    
    if the_set is None:
        raise ValueError("What is this I don't even")
    
    if the_set == 'train' or the_set.startswith('neldermead'):
        im_base = Path('MOTS') / 'ims'
    elif the_set == 'test':
        im_base = Path('MOTS') / 'test_ims'
    else:
        raise ValueError("What is this I don't even")
    
    txts = list(folder.glob('*.txt'))
    txts.sort()
    
    n_threads = min(len(txts), multiprocessing.cpu_count())
    
    vis_paths = [(vis_folder / f"{txt.stem}.mp4") for txt in txts]
    im_folders = [(im_base / txt.stem) for txt in txts]
    
    if len(sys.argv) == 3:
        other_run = sys.argv[-2]
        print("Main:", run_name, "Other:", other_run)
        other_folder = Path('output') / other_run
        other_txts = list(other_folder.glob('*.txt'))
        other_txts.sort()
        vis_folder = folder / f"vis_{other_run}"
        vis_folder.mkdir(exist_ok=True)
        vis_paths = [(vis_folder / f"{txt.stem}.mp4") for txt in txts]
        with multiprocessing.Pool(n_threads) as pool:        
            pool.starmap(visualize_tracks_images_compare, zip(txts, other_txts, vis_paths, im_folders))
        
    else:
#        for txt, vispath, imfolder in zip(txts, vis_paths, im_folders):
#            visualize_tracks_images(txt, vispath, imfolder)
        with multiprocessing.Pool(n_threads) as pool:        
            pool.starmap(visualize_tracks_images, zip(txts, vis_paths, im_folders))

if __name__ == '__main__':
    main()


