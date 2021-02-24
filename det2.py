""" A sane interface to Mask R-CNN in Detectron2 """

import numpy as np
from pathlib import Path
from datetime import timedelta, datetime
from random import choice
import gc

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import imageio as iio
from datetime import timedelta
from time import time

import full_format
from util import show, multi_glob
from datasets import get_kitti_mots_base, get_classes

def convert_kitti_class_to_mscoco(x):
    if x == 2:    # person
        return 0
    elif x == 1:  # car
        return 2
    elif x == 3:  # bike
        return 1
    else:
        raise ValueError(f"Incompatible class ID {x}")

class MaskRCNN:
    """
        weights: either None or a Path to a weights file
                 if None, default weights pretrained on MS COCO will be used
    """
    def __init__(self, weights):
        if weights is None:
            detectron2_identifier = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(detectron2_identifier))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron2_identifier)
        else:
            cfg_file = weights.parent / 'cfg.cfg'
            cfg = get_cfg()
            cfg.merge_from_file(str(cfg_file))
            cfg.MODEL.WEIGHTS = str(weights)
            
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        predictor = DefaultPredictor(cfg)
        
        self.predictor = predictor
        self.cfg = cfg
        
    def predict(self, im):
        # Detectron2 expects BGR for some reason, so we load with cv2 instead of iio
        return self.predictor(im)
        
    def visualize(self, im, outputs):
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]
    
    def save_objects(self, outputs, im_path, classes, out_path=None, out_path_suffix='det2', direct=False):
        if not direct:
            if out_path is None:
                out_folder = Path.resolve(im_path).parent.parent / out_path_suffix
                if not out_folder.is_dir():
                    out_folder.mkdir()
            else:
                out_folder = out_path
        else:
            # in case we have multiple classes
            out_pmasks = dict()
            out_boxes = dict()
        
        instances = outputs["instances"].to("cpu")
        pmasks_orig = instances._fields['pred_masks'].numpy()
        pclasses = instances._fields['pred_classes']
        pboxes_orig = instances._fields['pred_boxes']
        pscores_orig = instances._fields['scores'].numpy()
        
        n = pmasks_orig.shape[0]
        
        centers_orig = pboxes_orig.get_centers().numpy()
        
        for current_class in classes.keys():
            n_people = 0
            indices = []
            for i in range(n):
                if pclasses[i] == convert_kitti_class_to_mscoco(current_class):
                    n_people += 1
                    indices.append(i)
            
            centers = centers_orig[indices, :] # Convert to our indices, only people        
            pmasks = pmasks_orig[indices, ::]  # Same
            pscores = pscores_orig[indices]    # Same
            
            boxes = pboxes_orig.tensor.numpy()
            boxes = boxes[indices, ::] # Convert to our indices, only people
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            
            if direct:
                if len(classes) == 1:
                    return pmasks, boxes    
                else:
                    out_pmasks[current_class] = pmasks
                    out_boxes[current_class] = boxes
            else:
                if pmasks.shape[0] > 0:
                    text = full_format.encode(pmasks)
                else:
                    text = ""
            
                output = (out_folder / f"{im_path.stem}_{current_class}.txt")    
                output.write_text(text)
                
                boxes_path = output.with_suffix('.boxes')
                np.savetxt(boxes_path, boxes, delimiter=',', fmt='%5.2f')
                
        if direct:
            return out_pmasks, out_boxes

class CenterNetLite(MaskRCNN):
    def __init__(self, config, weights):
        from centermask.config import get_cfg

        cfg = get_cfg()
        cfg.merge_from_file(config)
        cfg.MODEL.WEIGHTS = weights

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = -1e6
        predictor = DefaultPredictor(cfg)

        self.predictor = predictor
        self.cfg = cfg

def exec_speed():
    weights = None #Path('output') / 'train_det2_2020_09_01__14_31_06' / 'model_0008399.pth'
    ops = 'det2_speed'
    
    m = MaskRCNN(weights=weights)
    
    folder = Path('MOTS') / 'train' 
    jpgs = list(folder.glob('*/img1/*.jpg'))
    
    dt_sum = 0
    
    N = 64
    
    for i in range(N):
        jpg = choice(jpgs)
        im = cv2.imread(str(jpg))
        
        prev = time()
        outputs = m.predict(im)
        pixel_nms(outputs['instances'])
        post = time()
        
        dt = post - prev
        dt_sum += dt
    
    dt_sum /= N
    
    print(timedelta(seconds=dt_sum))
    print(f"About {1/dt_sum} FPS")

def get_centernet_model():
    return CenterNetLite('./centermask2/configs/centermask/centermask_lite_V_39_eSE_FPN_ms_4x_fintue_mots.yaml',
                      './centermask2/output/centermask/CenterMask-Lite-V-39-ms-4x_fintune_mots/new_model_0079999.pth')

def main_centernet_lite():
    sys.path.append("./centermask2")
    m = get_centernet_model()
    run('MOTS', m, 'det2_centernet')


def main_maskrcnn():
    # Set to None for pretrained
    #weights = Path('trained_det2') / 'model_7500.pth'
    #ops = 'det2_trained'
    
    #weights = Path('output/train_det2_2020_09_09__08_21_08/model_0007499.pth')
    #ops = 'det2_val09'
    
    #weights = Path('output/train_det2_2020_09_09__14_09_00/model_0007499.pth')
    #ops = 'det2_val11'
    
    weights = None
    ops = 'det2'
    
    m = MaskRCNN(weights=weights)
    dataset = 'KITTI_train'
    run(dataset, m, ops)
    
    

def pixel_nms(instances):
    # torch.cuda.synchronize()
    # t0 = time()
    assert list(instances.scores) == sorted(instances.scores, reverse=True)
    pred_masks = instances.pred_masks
    
    if pred_masks.shape[0] == 0:
        return
        
    if True:
        from pixel_nms.pixel_nms import pixel_nms_cuda_  # Requires 'python setup.py build_ext --inplace' in pixel_nms/
        assert pred_masks.dtype is torch.bool, f"pred_masks is of type {pred_masks.dtype}, expected torch.bool. Size is {pred_masks.shape}"
        pixel_nms_cuda_(pred_masks)
    else:
        supress = torch.zeros(pred_masks[0].shape, dtype=torch.bool, device=pred_masks.device)
        for mask in pred_masks:
            mask[supress] = False
            supress[mask] = True
    # assert pred_masks.sum(0).max() <= 1
    # torch.cuda.synchronize()
    # print(time() - t0)

def get_seqs(dataset, explicit_seqnames=None):
    folders = []
    if dataset.startswith('MOTS'):
        if explicit_seqnames is None:
            seqs1 = [x.name for x in (Path(dataset) / 'train').glob('*') if x.is_dir()]
            seqs2 = [x.name for x in (Path(dataset) / 'test').glob('*') if x.is_dir()]
            seqs = seqs1 + seqs2
        else:
            seqs = explicit_seqnames
        
        for seq in seqs:
            train_folder = Path(dataset) / 'train' / seq / 'img1'
            test_folder = Path(dataset) / 'test' / seq / 'img1'

            if train_folder.is_dir():
                folder = train_folder
            elif test_folder.is_dir():
                folder = test_folder
            else:
                raise ValueError(f"Sequence {seq} is neither in the train nor test set")
            folders.append(folder)
    elif dataset.startswith('KITTI'):
        if dataset == 'KITTI_train':
            seqmap_path = (Path('KITTI') / 'train.seqmap')
            main_set = 'training'
        elif dataset == 'KITTI_val':
            seqmap_path= (Path('KITTI') / 'val.seqmap')
            main_set = 'training'
        elif dataset == 'KITTI_test':
            seqmap_path = (Path('KITTI') / 'test.seqmap')
            main_set = 'testing'
        else:
            raise ValueError(f"Unknown KITTI dataset {dataset}")
            
        if explicit_seqnames is None:
            seqmap = seqmap_path.read_text().split('\n')
            seqs = [x.split(' ')[0] for x in seqmap]
        else:
            seqs = explicit_seqnames
            
        base = get_kitti_mots_base() / 'data_tracking_image_2' / main_set / 'image_02'
        
        for seq in seqs:
            folder = base / seq
            if folder.is_dir():
                folders.append(folder)
            else:
                raise ValueError(f"Could not find {folder}")
    
    return folders

def run(dataset, m, ops):
    total_time = timedelta()
    total_frames = 0

    backuped = []
    
    classes = get_classes(dataset)

    # ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']: # test
    # ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']: # train
    for folder in get_seqs(dataset, explicit_seqnames=None):
        jpgs = multi_glob(folder, ['*.jpg', '*.png'])
        jpgs.sort()

        for jpg in jpgs:
            total_frames += 1
            
            im = cv2.imread(str(jpg))
            
            before = datetime.now()

            outputs = m.predict(im)
            pixel_nms(outputs['instances'])
            
            after = datetime.now()
            total_time += (after - before)
            
            #to_show = m.visualize(im, outputs)
            #cv2.imwrite(str(Path('lol')/jpg.name), to_show)
            
            out_path = None
            if dataset.startswith('KITTI'):
                out_path = jpg.parent.parent.parent / ops
                out_path.mkdir(exist_ok=True)
                
                out_path /= jpg.parent.name
                out_path.mkdir(exist_ok=True)
                
            m.save_objects(outputs, jpg, out_path_suffix=ops, out_path=out_path, classes=classes)
            print(ops, jpg)
    print(total_frames, "frames and total GPU time was", total_time)
    fps = total_frames/total_time.total_seconds()
    print("FPS:", fps)
    print("Backup used for:", backuped)

if __name__ == '__main__':
    import sys
    e = ValueError("Either run `python det2.py maskrcnn` or `python det2.py speed` or `python det2.py centernet`")
    if len(sys.argv) < 2:
        raise e
    elif sys.argv[1] == 'maskrcnn':
        main_maskrcnn()
    elif sys.argv[1] == 'speed':
        exec_speed()
    elif sys.argv[1] == 'centernet':
        main_centernet_lite()
    else:
        raise e

        

