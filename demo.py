""" An easy-to-use interface for running Detectron2 + SORTS on video files """

from det2 import MaskRCNN, pixel_nms
from sorts import Settings, SORTS, track_okay
import full_format
from visualize import draw_mask
from util import long_str, intr
from colors import get_colors

from pathlib import Path
import imageio as iio
import cv2
import numpy as np
import sys

def main(video_file=None):
    n_colors = 32
    colors = get_colors(n_colors)

    if video_file is None:
        raise ValueError("What is this I don't even")
    
    out_folder = Path('video_out')
    if not out_folder.is_dir():
        out_folder.mkdir(parents=True) 
    
    w_path = Path('trained_det2') / 'model_7500.pth'
    if w_path.is_file():
        print(f"Using weights from {w_path}")
    else:
        w_path = None
        print("Using pretrained Mask R-CNN weights")
    det2 = MaskRCNN(weights=w_path) 
    
    s = Settings()
    s.iou_mode = 'full'
    
    s.y_cutoff = 0.65
    s.iou_threshold = 0.3
    s.kf_R2 = 0.1712294370040106
    s.kf_P4 = 3171.190981805896
    s.kf_Pscale = 203.68549925089593
    s.kf_Qscale = 0.02772065175995274
    s.min_age = 0.13111967759057736
    s.max_age = 0.048721120733634465
    s.h_min = 0.036573430957952972
    
    s.use_reid = True
    s.reid_thresh = 0.8974024428596924
    s.reid_border = 0.009914818655492189
    s.reid_memory = 1.7392946816377515
    s.reid_storage = 1.7234346233287139
    s.reid_shortest = 0.16594656988159678
    s.reid_lookback = 0.16526851056957566
    s.reid_minheight = 0.07397759436969294
    
    s.explicit_name = 'direct_video'
    
    s.multi = False
    s.time = False
    
    vid_reader = iio.get_reader(video_file)
    md = vid_reader.get_meta_data()
    
    total_frames = vid_reader.count_frames()
    
    im_cache = dict()
    def get_im(frame_no):
        return im_cache[frame_no]
    
    fps = md['fps']
    sorts = SORTS(md['size'][1], md['size'][0], fps, get_im, settings=s)
    
    cap = cv2.VideoCapture(str(video_file))
    
    out_path = out_folder / video_file.name
    print(f"Writing to {out_path}")
    with iio.get_writer(out_path, fps=intr(fps*0.75)) as outvid:
        frame_no = 0
        while cap.isOpened():
            frame_no += 1
            ret, frame = cap.read()
            
            while len(im_cache) > 100:
                key = next(iter(im_cache.keys()))
                im_cache.pop(key)
            
            if not ret:
                break
            
            out = det2.predict(frame)
            pixel_nms(out['instances'])
            masks, boxes = det2.save_objects(outputs=out, im_path='', classes={2: 'person'}, direct=True)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_cache[frame_no] = frame
            
            trackers = sorts.update(boxes, masks, timing=None)
            l = [(t.orig_mask, t.id) for t in trackers if track_okay(t)]

            draw = frame.copy()
            for mask, track_id in l:
                color = colors[track_id % n_colors]
                draw = draw_mask(draw, mask, color, str(track_id))
            
            outvid.append_data(draw)
            print(frame_no, '/', total_frames)
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: `python demo.py VIDEO_FILE`')
        sys.exit(1)
        
    video_file = Path(sys.argv[1])
    
    if not video_file.is_file():
        raise FileNotFoundError(video_file)
        
    main(video_file=video_file)


