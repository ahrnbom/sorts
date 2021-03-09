"""
        SORTS: A Simple, Online and Realtime Tracker with Segmentation   
        By Martin Ahrnbom, martin.ahrnbom@math.lth.se
        
        based on 
        SORT: A Simple, Online and Realtime Tracker
        Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from pathlib import Path
import numpy as np
from filterpy.kalman import KalmanFilter
from datetime import datetime, timedelta
import imageio as iio
import multiprocessing
from random import random
import cv2
from io import StringIO

import full_format
from util import smallest_box, long_str, vector_similarity, intr, clip, show
from reid import ReID
from ini import read_ini
from timing import Timing
from datasets import get_seqs, get_kitti_mots_base

try:
    from numba import jit
except:
    def jit(func):
        return func

np.random.seed(0)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


@jit
def iou(bb_test, bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
  return(o)


def iou_mask(mask1, mask2, det1, det2, settings):
    
    if settings.iou_mode == 'box':
        return iou(det1, det2)
    
    # Check if there's any chance of intersection
    if (det1[0] > det2[2]) or (det2[0] > det1[2]) or \
       (det1[1] > det2[3]) or (det2[1] > det2[3]):     
        return 0.0
    
    # Crop a bunch of pointless zeroes    
    x1 = np.minimum(det1[0], det2[0]).astype(int)
    y1 = np.minimum(det1[1], det2[1]).astype(int)
    x2 = np.maximum(det1[2], det2[2]).astype(int)
    y2 = np.maximum(det1[3], det2[3]).astype(int)
    
    ytot = y2-y1
    y2 = y1 + int(settings.y_cutoff*ytot)
    
    mask1 = mask1[y1:y2, x1:x2]    
    mask2 = mask2[y1:y2, x1:x2]
    
    if settings.iou_mode == 'full':
        intersection = np.sum(np.logical_and(mask1, mask2))
        union = np.sum(np.logical_or(mask1, mask2))
        if union > 0:
            return float(intersection) / float(union)
        else:
            return 0.0
    elif settings.iou_mode == 'iom':
        intersection = np.logical_and(mask1, mask2)
        min_area = min(np.sum(mask1), np.sum(mask2))
        if min_area > 0:
            return float(np.sum(intersection)) / min_area
        else:
            return 0.0
            

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox, mask, frame_no, sorts, settings):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= settings.kf_R2
        self.kf.P[4:,4:] *= settings.kf_P4 #give high uncertainty to the unobservable initial velocities
        self.kf.P *= settings.kf_Pscale
        self.kf.Q[-1,-1] *= settings.kf_Qscale
        self.kf.Q[4:,4:] *= settings.kf_Qscale

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = None # Initialize as None, if it gets old enough it'll get an ID later
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.spawn_time = frame_no
        self.speed = 0 # pixels per frame
        
        self.mask = mask
        self.orig_mask = mask
        self.settings = settings
        self.sorts = sorts
        
        self.active_now = False
        
        if settings.use_reid:
            self.reid_masks = dict()
            self.reid_vector = None
            self.reid_frameno = None

    def update(self, bbox, mask, curr_frame_no, settings, timing=None):
        """
        Updates the state vector with observed bbox.
        """
        if settings.time:
            timing.start('track_update')
        
        # Update speed
        if self.history:
            prev = self.history[-1]
            prev_x = (prev[0] + prev[2])/2
            prev_y = (prev[1] + prev[3])/2
            
            curr_x = (bbox[0] + bbox[2])/2
            curr_y = (bbox[1] + bbox[3])/2
            
            distance = np.sqrt( (curr_x - prev_x)**2 + (curr_y - prev_y)**2 )
            curr_speed = distance / self.time_since_update
            
            self.speed = 0.9*self.speed + 0.1*curr_speed
            
        self.time_since_update = 0
        self.active_now = True    
        
        # Note: SORT actually clears the history here for some reason
        self.history.append(bbox)
        
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        
        self.mask = mask
        self.orig_mask = mask
        
        if settings.time:
            timing.stop('track_update')

        if (self.id is None):
            
            # At the very start of sequence, we accept "too young" tracks
            if (len(self.history) >= self.sorts.min_age) or (self.spawn_time <= self.sorts.min_age):
                # Assign track ID 
                
                new_id = None
                try_reid = settings.use_reid
                
                # Some early checks to not even try ReID 
                if try_reid:
                    # If there are no waiting tracks, no point in trying
                    if not self.sorts.reid_list:
                        try_reid = False
                        
                    # Check if track is too small for ReID to be reliable
                    _, y1, _, y2 = bbox
                    if (y2-y1) <= (settings.reid_minheight * self.sorts.im_height):
                        try_reid = False
                        
                if try_reid:

                    im = self.sorts.get_im(curr_frame_no)
                    if settings.time:
                        timing.start('reid_assign')
                    v, success = self.sorts.reid.get_vector(im, self.mask)
                    
                    if success:
                        best = settings.reid_thresh
                        best_i = None
                        for i, m in enumerate(self.sorts.reid_list):  
                            if settings.reid_spatial is not None:
                                # Check if track is spatially feasible
                                reid_age = curr_frame_no - m.reid_frameno
                                possible_distance = reid_age * m.speed * settings.reid_spatial
                                
                                old_pos = m.history[-1]
                                old_x = (old_pos[0] + old_pos[2])/2
                                old_y = (old_pos[1] + old_pos[3])/2
                                
                                new_x = (bbox[0] + bbox[2])/2
                                new_y = (bbox[1] + bbox[3])/2
                                
                                distance = np.sqrt( (old_x-new_x)**2 + (old_y-new_y)**2 )
                                
                                if distance >= possible_distance:
                                    # Ignore this ReID possibility
                                    continue
                        
                            similarity = vector_similarity(v, m.reid_vector)
                            if similarity >= best:
                                best = similarity
                                new_id = m.id
                                best_i = i
                    
                        # If we found a match, then that track should get removed
                        # as to not match any future tracks
                        if best_i is not None:
                            self.sorts.reid_list.pop(best_i)
                    if settings.time:
                        timing.stop('reid_assign')
                        
                
                # Generate a new Track ID if nothing suggests otherwise
                if new_id is None:
                    new_id = KalmanBoxTracker.count
                    KalmanBoxTracker.count += 1
                
                self.id = new_id
                
                
    
    def predict(self, curr_frame_no):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.active_now = False
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.update_mask(curr_frame_no)
        
        new_box = convert_x_to_bbox(self.kf.x)
        
        return new_box, self.mask

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
    
    def update_mask(self, curr_frame_no):

        if len(self.history) >= 2:
            old_box = self.history[-2]
            new_box = self.history[-1]
            new_x = (new_box[0] + new_box[2])/2
            new_y = (new_box[1] + new_box[3])/2
            old_x = (old_box[0] + old_box[2])/2
            old_y = (old_box[1] + old_box[3])/2
            
            dx = new_x - old_x
            dy = new_y - old_y
            
            self.orig_mask = self.mask # Shouldn't need copy, since np.roll only returns a new view
            self.mask = np.roll(self.mask, intr(dy), axis=0)
            self.mask = np.roll(self.mask, intr(dx), axis=1)
                
            if self.settings.use_reid:
                last_mask = curr_frame_no - self.time_since_update
                if not last_mask in self.reid_masks:
                    self.reid_masks[last_mask] = self.orig_mask

                # Remove any masks that are too old
                for frame_no in list(self.reid_masks.keys()):
                    if curr_frame_no - frame_no > self.sorts.reid_memory:
                        self.reid_masks.pop(frame_no)
        else:
            # Cannot estimate movement without two frames
            return
    
    def compute_reid(self, get_im, reid, curr_frame_no, timing=None):
        # Pick a good time to compute the ReID vector. Not too long ago, but also
        # preferrably not the last frame (since it could be partially occluded)
        available = list(self.reid_masks.keys())
        
        if len(available) >= self.sorts.reid_lookback:
            frame_no = available[-self.sorts.reid_lookback]
        else:
            frame_no = available[len(available)//2]
            # Middle of the track is probably the safest bet in this situation
        
        mask = self.reid_masks[frame_no]
        if self.settings.time:
            timing.stop('reid_storing', whatever=True)
        im = get_im(frame_no)
        if self.settings.time:
            timing.start('reid_storing')
        v, success = reid.get_vector(im, mask)
        
        if success:
            self.reid_vector = v
            self.reid_frameno = curr_frame_no
        
def associate_detections_to_trackers(detections, trackers, masks, trackers_masks, settings):
    """
    Assigns detections to tracked object (both represented as bounding boxes and masks)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    centers = np.vstack([(detections[:,0] + detections[:,2])/2, 
                         (detections[:,1] + detections[:,3])/2]).T
    trackers_centers = np.vstack([(trackers[:,0] + trackers[:,2])/2,
                                  (trackers[:,1] + trackers[:,3])/2]).T

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou_mask(masks[d,::], trackers_masks[t], det, trk, settings)  

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > settings.iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<settings.iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Settings(dict):
    def __init__(self, **kwargs):
        # Start with defaults
        # All keys should use only letters and underscores!
        self['iou_mode'] = 'full' # Valid options: 'full', 'iom', 'box'
        self['y_cutoff'] = 1.0 # 0 - cut everything, 1 - cut nothing
        self['iou_threshold'] = 0.3 # Between 0 and 1
        self['set'] = 'train' # Valid options: 'train', 'test', 'neldermead_train', 'neldermead_val'
        self['set_folder'] = None # This value is automatically deduced from 'set'
        self['kf_R2'] = 10. # SORT: 10.
        self['kf_P4'] = 1000. # SORT: 1000.
        self['kf_Pscale'] = 10. # SORT: 10.
        self['kf_Qscale'] = 0.01 # SORT: 0.01
        self['debug_prob'] = 0
        self['explicit_name'] = ""
        self['min_age'] = 0.1 # 0 to inf, in seconds (how many detections we need to "count" the track)
        self['max_age'] = 0.03 # 1 to inf, in seconds (how many frames without detections we keep a track)
        self['h_min'] = 0.035
        
        self['use_reid'] = True
        self['reid_thresh'] = 0.9
        self['reid_border'] = 0.01
        self['reid_memory'] = 1.7
        self['reid_storage'] = 1.7
        self['reid_shortest'] = 3.3
        self['reid_lookback'] = 0.17
        self['reid_minheight'] = 0.05
        self['reid_spatial'] = None # None - do not apply any spatial limit, otherwise set to a numerical value like 1.2
        
        self['multi'] = True
        self['dataset'] = 'MOTS'
        self['time'] = False
        self['detections'] = 'det2'
        
        self['obj_class'] = None # Integer for the class ID, or None for all available classes
        
        # Add any additional stuff
        keys = self.keys()
        for key, value in kwargs.items():
            if key in keys:
                self[key] = value
            else:
                raise ValueError(f"Construction error! Incorrect key {key}")
    
    def get_run_name(self):
        s = 'sorts__'
        for key, val in self.items():
            if not (key in ('debug_prob', 'explicit_name')):
                s += f"{key}-{val}__"
        s = s[:-2]
        
        if self['debug_prob'] > 0:
            if self['explicit_name']:
                return 'GARBAGE_' + self['explicit_name']        
            else:
                return 'GARBAGE_' + s
        
        if self['explicit_name']:
            return self['explicit_name']
        
        return s
    
    def summary(self):
        return '\n'.join([f"{key}: {val}" for key,val in self.items()])
            
    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            raise ValueError(f"Thing not found in settings: {name}")
    
    def __setattr__(self, name, value):
        if name in self.keys():
            self[name] = value
        else:
            raise ValueError(f"Tried to set {name} to {value} but key {name} does not exist")
    
    def __getstate__(self):
        state = dict()
        for name in self.keys():
            state[name] = self[name]
        return state
    
    def __setstate__(self, state):
        for name in state.keys():
            self[name] = state[name]

class SORTS(object):
    def __init__(self, im_height, im_width, fps, get_im, settings=Settings()):
        """
        Sets key parameters for SORT
        """
        self.trackers = []
        self.frame_count = 0
        
        self.settings = settings
        
        self.im_height = im_height
        self.im_width  = im_width
        self.fps = fps
        self.get_im = get_im
        
        def oneplus(x):
            # Integer, at least one 
            return clip(intr(x), 1, float("inf"))
        
        self.min_age = oneplus(settings.min_age * fps)
        self.max_age = oneplus(settings.max_age * fps)
        
        if settings.use_reid:
            self.reid_list = []
            self.reid = ReID()
            
            self.reid_storage = oneplus(settings.reid_storage * fps)
            self.reid_memory = oneplus(settings.reid_memory * fps)
            self.reid_lookback = oneplus(settings.reid_lookback * fps)
            self.reid_shortest = oneplus(settings.reid_shortest * fps)
        
    def store_for_reid(self, track, curr_frame_no, timing=None):
        # Ignore too short tracks
        if len(track.history) < self.reid_shortest:
            return
        
        # Ignore tracks with no ID
        if track.id is None:
            return
        
        # Check if the track died near the borders. If so, ignore it.
        last_pos = track.history[-1]
        x1, y1, x2, y2 = last_pos
        
        border = self.im_height * self.settings.reid_border
        
        if (x1 <= border) or (y1 <= border) or (x2 >= self.im_width - border) or (y2 >= self.im_height - border):
            
            if ((y1 <= border) and (y2 >= self.im_height - border)):
                # If the track touches both the top and the bottom, then it's
                # just close to the camera, and we want to keep that one
                pass
            else:
                # This means the track is probably walking out of the video frame
                return
       
        # If track is small, ReID work poorly
        if (y2-y1) < self.settings.reid_minheight * self.im_height:
            return
       
        # Okay, so track seems like it could be connected using ReID.
        track.compute_reid(self.get_im, self.reid, curr_frame_no, timing=timing)
        if track.reid_vector is not None:
            self.reid_list.append(track)
       
    def update(self, dets=np.empty((0, 5)), masks=None, timing=None):
        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            masks - numpy array with shape (n, h, w). 
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        if masks is None:
            n_dets = dets.shape[0]
            masks = np.empty((n_dets, 1, 1))
        
        self.frame_count += 1
        
        if self.settings.time:
            timing.start('reid_cleanup')
        # Remove any tracks in ReID list, if they're too old
        if self.settings.use_reid:
            to_del = []
            for i, track in enumerate(self.reid_list):
                dt = self.frame_count - track.reid_frameno 
                if dt > self.reid_storage:
                    to_del.append(i)
            for i in reversed(to_del):
                self.reid_list.pop(i)
        if self.settings.time:
            timing.stop('reid_cleanup')
        
        if self.settings.time:
            timing.start('predicting')    
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(self.frame_count)[0][0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if self.settings.time:
            timing.stop('predicting')
        
        if self.settings.time:
            timing.start('associate')
        tracker_masks = [trk.mask for trk in self.trackers]
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,masks,tracker_masks,self.settings)
        if self.settings.time:
            timing.stop('associate')

        # update matched trackers with assigned detections

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], masks[m[0], ::], self.frame_count, self.settings, timing)
        
        if self.settings.time:
            timing.start('birth')    
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:], masks[i,::], self.frame_count, self, self.settings)
            
            if self.frame_count == 1:
                if self.settings.time:
                    timing.stop('birth')
                # Just to make sure tracks on the first frame have IDs
                trk.update(dets[i,:], masks[i,::], self.frame_count, self.settings, timing)
                if self.settings.time:
                    timing.start('birth')
            
            self.trackers.append(trk)
        if self.settings.time:
            timing.stop('birth')
                
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                if self.settings.use_reid:
                    if self.settings.time:
                        timing.start('reid_storing')
                    self.store_for_reid(self.trackers[i], self.frame_count, timing=timing)
                    if self.settings.time:
                        timing.stop('reid_storing')
                self.trackers.pop(i)
    
        return self.trackers


def track_okay(t):
    return (t.id is not None) and (t.id < 1000) and (t.active_now)


def process(frame, mot_tracker, dets, masks, out_file, s, timing=None):
    total_instances = 0
    
    id_prefix = 2000 # The '2' in front means 'pedestrian', used in MOTS dataset
    # if s.obj_class should be taken into account here, it should be a non-negative integer
    # otherwise we default to MOTS' standard of always using 2000
    if isinstance(s.obj_class, int) and s.obj_class >= 0:
        id_prefix = 1000*s.obj_class
    
    time_before = datetime.now()
    if s.time:
        timing.start('preprocessing')
    masks, dets = preprocess(masks, dets, s)
    if s.time:
        timing.stop('preprocessing')
    trackers = mot_tracker.update(dets, masks, timing)
    time_after = datetime.now()
    ex_time = (time_after - time_before) # subtraction creates timedelta objects!
    
    if s.time:
        timing.start('encoding_masks')
    
    l = [(t.orig_mask, t.id) for t in trackers if track_okay(t)]
    masks = [x[0] for x in l]
    ids = [x[1] for x in l]
    
    class_id = s.obj_class
    
    for i,mask in enumerate(masks):
        mask_str = full_format.encode_one(mask)
        h, w = mask.shape
    
        new_id = id_prefix + ids[i] 
        print(f"{frame} {new_id} {class_id} {h} {w} {mask_str}", file=out_file)
        
        total_instances += 1
    if s.time:
        timing.stop('encoding_masks')
    return ex_time, total_instances


""" Takes masks from full_format and reworks them to detections and masks in 
    the format expected by the update function; (N, H, W) instead of (H, W, N)
    Also removes detections that are too small
"""
def preprocess(masks, dets, settings):
    masks = np.moveaxis(masks, -1, 0)
    
    n, h, w = masks.shape
    if n > 0:
        h_thresh = h * settings.h_min
        
        if n == 1 and len(dets.shape) == 1:
            dets = np.expand_dims(dets, axis=0)
        
        good = np.ones((n,), dtype=bool)
        for i in range(n):
            box = dets[i, :]
            
            box_h = box[3] - box[1]
            if box_h < h_thresh:
                good[i] = False
        
        dets = dets[good, :]
        masks = masks[good, ::]
    else:
        masks = np.empty((0,0,0), dtype=bool)
        dets = np.empty((0,5), dtype=np.float32)

    return masks, dets


def run(s, seq): 
    if s.time:
        timing = Timing()
    else:
        timing = None

    run_name = s.get_run_name()
    out_folder = Path("output") / run_name

    seq_frames = 0
    total_instances = 0
        
    seqinfo = read_ini(s.dataset, seq)
    h = seqinfo['imHeight']
    w = seqinfo['imWidth']
    fps = seqinfo['frameRate']
    
    cache = {}
    cache_max = 16
    other = {'time_discount': timedelta(), 'cache_uses': 0}
    
    def get_im(frame_no):
        if s.time:
            timing.start('image_loading')
        while len(cache) > cache_max:
            # Delete oldest
            key = next(iter(cache.keys()))
            cache.pop(key)
        
        if s.dataset.startswith('MOT'):    
            impath = Path(s.dataset) / s.set_folder / seq / 'img1' / f"{long_str(frame_no)}.jpg"
        elif s.dataset == 'KITTI':
            impath = get_kitti_mots_base() / 'data_tracking_image_2' / s.set_folder / 'image_02' / seq / f"{long_str(frame_no)}.png"
        else:
            raise ValueError(f"Not sure how to handle dataset {s.dataset} for image loading")
            
        if impath in cache:
            if s.time:
                timing.stop('image_loading')
            other['cache_uses'] += 1
            return cache[impath]
        else:
            before = datetime.now()
            im = iio.imread(impath)
            after = datetime.now()
            other['time_discount'] += (after - before)
            cache[impath] = im
            if s.time:
                timing.stop('image_loading')
            return im
    
    mot_tracker = SORTS(h, w, fps, get_im, settings=s)
    KalmanBoxTracker.count = 0 # Reset ID count
    
    suffix = 'txt'
    if s.debug_prob > 0:
        suffix = 'garbage'
    
    out_path = out_folder / f"{long_str(seq.split('-')[-1],4)}.{suffix}"
    
    if s.dataset.startswith('MOT'):
        folder = Path(s.dataset) / s.set_folder / seq / s.detections
    elif s.dataset == 'KITTI':
        folder = get_kitti_mots_base() / 'data_tracking_image_2' / s.set_folder / s.detections / seq
    
    total_time = timedelta()
    total_frames = 0
    
    with open(out_path, 'w') as out_file:    
        if s.obj_class is None:
            search = '*.txt'
        else:
            search = f"*_{s.obj_class}.txt"
            
        txts = list(folder.glob(search))
        txts.sort()
        
        for txt in txts:
            frame = int(txt.stem.split('_')[0])
            total_frames += 1
            seq_frames += 1
            
            if s.time:
                timing.start('reading_files')
            masks = full_format.read_file(txt, dtype=bool)
            boxes = txt.with_suffix('.boxes').read_text()
            if boxes:
                dets = np.genfromtxt(StringIO(boxes), delimiter=',')
            else:
                dets = np.empty((0,5), dtype=np.float32)
                
            if s.time:
                timing.stop('reading_files')
            ex_time, instances = process(frame, mot_tracker, dets, masks, out_file, s, timing)
            total_time += ex_time
            total_instances += instances
            if frame % 20 == 0:
                print(run_name, txt)
    
    print(seq, 'Time:', total_time, 'Image loading discount:', other['time_discount'])
    if s.use_reid:
        print('Cache uses:', other['cache_uses'])
        print('ReID failures:', mot_tracker.reid.fails_count)
        print('ReID total vectors:', mot_tracker.reid.total_vectors)
        print("Total instances", total_instances)
    
    return total_time - other['time_discount'], total_frames, timing

def main(s=None):
    if s is None:
        raise ValueError('No settings provided!')
    
    run_name = s.get_run_name()
    print(f"Run name: {run_name}")
    
    total_time = timedelta()
    total_frames = 0
    
    if s.time:
        total_timing = Timing()
    
    out_folder = Path("output") / run_name
    if not out_folder.is_dir():
        out_folder.mkdir(parents=True)

    # Putting them in this order guarantees that set_folder is set before writing to file, which can be used for visualizations or whatever
    seqs = get_seqs(s)        
    (out_folder / 'this_run.settings').write_text(s.summary())
    
    if s.multi:
        n = len(seqs)
        with multiprocessing.Pool(n) as pool:
            res = pool.starmap(run, zip([s]*n, seqs))
            for this_time, this_frames, timings in res:
                total_time += this_time
                total_frames += this_frames
                if s.time:
                    total_timing += timings
    else:
        for seq in seqs:
            this_time, this_frames, timings = run(s, seq)
            total_time += this_time
            total_frames += this_frames
            if s.time:
                total_timing += timings
                
    time_text = f"Total execution time: {total_time}, Total frames: {total_frames}, FPS: {total_frames/total_time.total_seconds()}"
    print(time_text)
    (out_folder / 'execution.time').write_text(time_text)
    
    if s.time:
        with (out_folder / 'timing.timing').open('w') as f:
            total_timing.print_totals(to_file=f)
        print("Timing total time:",total_timing.total_time())
        print("Total frames:", total_frames)

