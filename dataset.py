import os
import numpy as np
import json 
from pyquaternion import Quaternion
from abc import ABC, abstractmethod
import cv2
from copy import deepcopy 
from shapely.geometry import Polygon 






class Box:
    """  Simple data class representing a 3d Box including label, score and velocity """

    def __init__(self,
                center,
                size,
                orientation,
                label,
                score,
                velocity,
                name,
                token
                ):
                """
                : param center: Center of box given as x,y, z
                : param size : Size of box in width, length, height.
                : param orientation: Box orientation.
                : param label : Integer label optional.
                : param score : Classification score, optional.
                : param velocity : Box velocity in x, y, z direction.
                : param name : Box name, optional. Can be used e.g. for denote category name.
                : token: Unique string Identifier from DB.
                """

                assert not np.any(np.isnan(center))
                assert not np.any(np.isnan(size))
                assert len(center) == 3
                assert len(size) == 3
                assert type(orientation) == Quaternion 


                self.center = np.array(center)
                self.wlh = np.array(size)
                self.orientation = orientation
                self.label = int(label) if not np.isnan(label) else label 
                self.score = float(score) if not np.isnan(score) else score 
                self.velocity = np.array(velocity)
                self.name = name 
                self.token = token 


    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or 
         (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel
    
    def __repr__(self):
        repr_str = 'label : {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f},{:.2f}, {:.2f}],' \
                  'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, '\
                      'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1],
        self.center[2], self.wlh[0], self.wlh[1], self.wlh[2], self.orientation.axis[0], 
        self.orientation.axis[1], self.orientation.axis[2], self.orientation.degrees,
        self.orientation.radians, self.velocity[0], self.velocity[1], self.velocity[2],
        self.name, self.token)

    @property
    def rotation_matrix(self):
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The boxes rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x):
        """
        Applies a translation.
        : param x : np.float: 3, 1>. Translation in x, y, z direction.
        """

        self.center += x

    def rotate(self, quaternion):
        """
        Rotates box.
        : param quaternion: Rotation to apply.
        """

        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation 
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor=1.0):
        """
        return the bounding box corners.
        : param wlh_factor : Multiply w,l,h by a factor to scale the box.
        : return : <np.float: 3, 8>. First four corners are the ones facing forwards.
        last the ones facing backwards.
        """

        w, l, h = self.wlh * wlh_factor 

        # 3D bounding box corners. ( x point forwards, y to the left , z up)

        x_corners = l / 2 *  np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1,-1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        
        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate 
        x,y,z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1,  :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self):
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forwards, last two face 
        backward.
        """

        return self.corners()[:, [2,3,7,6]]


    def view_points(points, view, normalize):
        """
        :param points: <np.ndarray (3, n)>
        :param view : <np.ndarray: (n. n)> Defines an arbitraty projection.
                      (n<=4)
        :param normalize: bool , whether to normaliz the remaining coordinate (along the third axis)
        : return : <np.float32: 3, n>. Mapped point. if Normalize = False the third coordinate is the height.
        """

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4                 
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view 

        nbr_points = points.shape[1]

        # Do operation in homogeneous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points






    def render(self,
                axis,
                view,
                normalize,
                colors,
                linewidth):
                """
                Renders the box in the provided matplotlib axis.
                : param axis: Axis onto which the box should be drawn.
                : param view: <np.array: 3,3>. Define a projection in needed(e.g. for dawing
                projection in an image).
                : param normalize: whether to normalize the remaining coordinate.
                : param colors : (<Matplotlin.colors>: 3). Valid Matplotlib colors (Mstr> or 
                normalized RGB tuple) for front, back and sides.
                : param linewidth: Width in pixel of the box sides.
                """

                corners = self.view_points(self.corners(),view, normalize=normalize)[:2, :]
                
                def draw_rect(selected_corners, color):
                    prev = selected_corners[-1]
                    for corner in selected_corners:
                        axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                        prev = corner 

                # Draw the sides 
                for i in range(4):
                    axis.plot([corners.T[i][0], corners.T[i+4][0]], 
                              [corners.T[i][1], corners.T[i+4][1]],
                              color=colors[2], linewidth=linewidth)
                    
                # Draw fron first 4 corners and rear (last 4 corners) rectangels(3d)/2d
                draw_rect(corners.T[:4], colors[0])
                draw_rect(corners.T[4:], colors[1])
                 
            

      

                # Draw line indicating the front 
                center_bottom_forward = np.mean(corners.T[2:4], axis=0)
                center_bottom = np.mean(corners.T[[2,3,7,6]], axis=0)
                axis.plot([center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0], linewidth=linewidth)

    def render_cv2(self,
                   im,
                   view,
                   normalize,
                   colors,
                   linewidth):
                   """
                   Renders box using OpenCV2
                   :param im : <np.array: width, height, 3>. Image array . Channels are in BGR order
                   :param view: <np.array: 3, 3>. Define a projection if needed (eg. for drawing projectijon in an image).
                   : param normalize : whether to normalize the remaining coordinate.
                   : param colors : (( R, G, B), (R, G, B), (R, G, B)). Colors for front, side and rear.
                   : param linewidth: Linewidth for plot.
                   """
                   corners = self.view_points(self.corners(), view, normalize=normalize)[:2, :]

                   def draw_rect(selected_corners, color):
                       prev = selected_corners[-1]
                       for corner in selected_corners:
                           cv2.line(im,(int(prev[0]), int(prev[1])),(int(corner[0]), int(corner[1])),color, linewidth)
                           prev = corner

                    # Draw the sides
                   for i in range(4):
                        cv2.line(im,
                            (int(corners.T[i][0]), int(corners.T[i][1])),
                            (int(corners.T[i+4][0]), int(corners.T[i+4][1])),
                            colors[2][::-1], linewidth)

                      

                    # Draw front first four corners and rear(last 4 corners) rectangels(3d)/lines(2d)
                   draw_rect(corners.T[:4], colors[0][::-1])
                   draw_rect(corners.T[4:], colors[1][::-1])

                   # Draw line indicating the front 
                   center_bottom_forward = np.mean(corners.T[2:4], axis=0)
                   center_bottom = np.mean(corners.T[[2,3,7,6]], axis=0)

                   cv2.line(im,
                             (int(center_bottom[0]), int(center_bottom[1])),
                            (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                            colors[0][::-1], linewidth
                            )

    def copy(self):
        """
        Create a copy of self.
        : return: A copy.
        """

        return copy.deepcopy(self)

    


class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x  # center x
        self.y = y  # center y
        self.z = z  # center z
        self.h = h  # height
        self.w = w  # width
        self.l = l  # length 
        self.o = o  # orientation
        self.s = None # score 
        
        
    def __str__(self):
        return "x: {}, y: {}, z: {}, heading: {}, height: {}, width: {}, length: {}, score: {}".format(self.x, self.y, self.z, self.o, self.h, self.w, self.l, self.s)
    
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z, 'center_heading':bbox.o, 'height': bbox.h, 'width': bbox.w, 'length':bbox.l}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])
        
    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
            
        
        return bbox
    
    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.o = data['heading']
        bbox.l = data['length']
        bbox.w = data['width']
        bbox.h = data['heigth']
        
        if 'score'   in data.keys():
            bbox.s = data['score']
            
        return bbox 
    
    
    @classmethod
    def copy_bbox(cls, bbox_a, bbox_b):
        """ bbox_b to bbox_a"""
        
        bbox_a.x = bbox_b.x 
        bbox_a.y = bbox_b.y 
        bbox_a.z = bbox_b.z   
        bbox_a.o = bbox_b.o   
        bbox_a.l = bbox_b.l 
        bbox_a.w = bbox_b.w 
        bbox_a.h = bbox_b.h 
        bbox_a.s = bbox_b.s  
        return 
    
    
    @classmethod
    def box2corners2d(cls, bbox):
        """
           the coordinates for bottom corners.
        """
        
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        
        pc0 = np.array([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,  bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2, bbox.z - bbox.h / 2])
        pc1 = np.array([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2, bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2, bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1 
        
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]
    
    @classmethod 
    def box2corners3d(self, bbox):
        """ the coordinates for bottom corners"""
        
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        
        return corners.tolist()
    
    
    @classmethod 
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        
        return result 
    
    
    
    @classmethod 
    def set_bbox_size(cls, bbox, size_array):
        result  = deepcopy(bbox) 
        
        result.l , result.w, result.h = size_array 
        
        return result 
    
    
    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] += state_array[4:]
        
        bbox = BBox.array2bbox(prev_array)
        
        return bbox 
    
    
    @classmethod 
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs, np.ones(pcs.shape[0])[:, np.newaxis]), axis=1)
        new_pcs = ego_matrix @ new_pcs.T 
        new_pcs = new_pcs.T[:, :3]
        
        return new_pcs 
    
    
    @classmethod 
    def edge2yaw(cls, center, edge):
        vec = edge - center    
        
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        
        if vec[1] < 0:
            yaw = - yaw 
            
        return yaw 
    
    @classmethod 
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1] / 2)
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])
        
        result = deepcopy(box)
        result.x, result.y, result.z = center          
        result.o = yaw 
        return result 
    
    
    
    
    
# data_utils

___all__ = ['inst_fileter', 'str2int', 'box_wrapper', 'type_filter', 'id_transform']

def str2int(strs):
    result = [int(s) for s in strs]
    return result 

def box_wrapper(bboxes, ids):
    frame_num = len(ids)
    result = list()
    for _i in range(frame_num):
        frame_result = list() 
        num = len(ids[_i])
        for _j in range(num):
            frame_result.append((ids[_i][j], bboxes[_i][j]))
        result.append(frame_result)
        
    return result 

def id_transform(ids):
    frame_num = len(ids)
    
    id_list = list()
    for _i in range(frame_num):
        id_list.append(ids[_i])
    id_list = sorted(list(set(id_list)))
    
    id_mapping = dict()
    for _i, id in enumerate(id_list):
        id_mapping[id] = _i 
        
    result = list()
    for _i in range(frame_num):
        frame_ids = list()
        frame_id_num = len(ids[_i])
        for _j in range(frame_id_num):
            frame_ids.append(id_mapping[ids[_i][_j]])
        result.append(frame_ids)
    return result 

def inst_filter(ids, bboxes, types, type_field=[1], id_trans=False):
    """ filter the bboxes according to types.
    """
    frame_num = len(ids)
    if id_trans:
        ids = id_transform(ids)
    id_result, bbox_result = [], []
    for _i in range(frame_num):
        frame_ids = list()
        frame_bboxes = list()
        frame_id_num = len(ids[_i])
        for _j in range(frame_id_num):
            obj_type = types[_i][_j]
            if obj_type in type_field:
                frame_ids.append(ids[_i][_j])
                frame_bboxes.append(BBox.array2bbox(bboxes[_i][_j]))
        id_result.append(frame_ids)
        bbox_result.append(frame_bboxes)
    return id_result, bbox_result 

def type_filter(contents, types, type_field=[1]):
    frame_num = len(types)
    content_result = [list() for i in range(len(type_field))]
    for _k , inst_type in enumerate(type_field):
        for _i in range(frame_num):
            frame_contents = list()
            frame_id_num = len(contents[_i])
            for _j in range(frame_id_num):
                if type[_i][_j] != inst_type :
                    continue  
                frame_contents.append(contents[_i][_j])
            content_result[_k].append(frame_contents)
    return content_result 


# preprocessing 

class BBoxCoarseFilter:
    def __init__(self, grid_size, scaler=100):
        self.gsize = grid_size 
        self.scalar = 100
        self.bbox_dict = dict()
        
    def bboxes2dict(self, bboxes):
        for i, bbox in enumerate(bboxes):
            grid_keys = self.compute_bbox_key(bbox)
            for key in grid_keys :
                if key not in  self.bbox_dict.keys():
                    self.bbox_dict[key] = set([i])
                else:
                    self.bbox_dict[key].add(i)
        return 
    
    def compute_bbox_key(self, bbox):
        corners = np.asarray(BBox.box2corners2d(bbox))
        min_keys = np.floor(np.min(corners, axis=0) / self.gsize).astype(np.int)
        max_keys = np.floor(np.max(corners, axis=0) / self.gsize).astype(np.int)
        
        # enumerate all the corners 
        grid_keys = [
            self.scalar * min_keys[0] + min_keys[1],
            self.scalar * min_keys[0] + max_keys[1],
            self.scalar * max_keys[0] + min_keys[1],
            self.scalar * max_keys[0] + max_keys[1]
        ]   
        return grid_keys 
    
    def related_bboxes(self, bbox):
        """
        return the list of related bboxes
        """
        
        result = set()
        grid_keys = self.compute_bbox_key(bbox)
        for key in grid_keys:
            if key in self.bbox_dict.keys():
                result.update(self.bbox_dict[key])
        return list(result)
    
    def clear(self):
        self.bbox_dict = dict()
        


def iou3d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap_area = reca.intersection(recb).area 
    iou_2d = overlap_area / (reca.area + recb.area - overlap_area)
    
    ha, hb = box_a.h, box_b.h 
    za, zb = box_a.z, box_b.z      
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    overlap_volume = overlap_area * overlap_height 
    union_volume = box_a.w * box_a.l * ha + box_b.w * box_b.l *hb - overlap_volume 
    iou_3d = overlap_volume / (union_volume + 1e-5)
    
    return iou_2d, iou_3d 

def weired_bbox(bbox):
    if bbox.l <= 0 or bbox.w <=0 or bbox.h  <= 0 :
        return True 
    else:
        return False   
    
    

def nms(dets, inst_types, threshold_low=0.1, threshold_high=0.5, threshold_yaw=0.3):
    """
    keep the bounding boxes with overlap <= threshold.
    """
    
    dets_coarse_filter = BBoxCoarseFilter(grid_size=100, scaler=100)
    dets_coarse_filter.bboxes2dict(dets)
    scores = np.asarray([det.s for det in dets])
    yaws = np.asarray([det.o for det in dets])
    order = np.argsort(scores)[::-1]
    
    
    result = list()
    result_types = list()  
    while order.size > 0:
        index = order[0]
        
        if weired_bbox(dets[index]):
            order = order[1:]
            continue    
        
        # locate the related bboxes 
        filter_indexes = dets_coarse_filter.related_bboxes(dets[index]) 
        in_mask = np.isin(order, filter_indexes) 
        related_idxes = order[in_mask]          
        related_idxes = np.asarray([i for i in related_idxes if inst_types[i] == inst_types[index]])
        
        # compute the ious 
        bbox_num = len(related_idxes)
        ious = np.zeros(bbox_num)
        for i,  idx in enumerate(related_idxes):
            ious[i]  = iou3d(dets[index], dets[idx])[1]
        related_inds = np.where(ious > threshold_low)
        related_inds_vote = np.where(ious > threshold_high)
        order_vote = related_idxes[related_inds_vote]
        
        if len(order_vote) >= 2 :
            # keep the bboxes with similar yaw 
            if order_vote.shaoe[0] <= 2 :
                score_index = np.argmax(scores[order_vote])
                median_yae = yaws[order_vote][score_index]
            elif order_vote.shape[0] % 2 == 0:
                tmp_yaw = yaws[order_vote].copy() 
                tmp_yaw = np.append(tmp_yaw, yaws[order_vote][0])
                median_yaw = np.median(tmp_yaw)
        else:
            meadian_yaw = np.median(yaws[order_vote])
        yaw_vote = np.where(np.abs(yaws[order_vote] - median_yaw) % ( 2 * np.pi) < threshold_yaw)[0]
        order_vote = order_vote[yaw_vote]
        
        # start weighted voting 
        vote_score_sum = np.sum(scores[order_vote])
        det_array =list()
        for idx in order_vote:
            det_array.append(BBox.bbox2array(dets[idx])[np.newaxis, :])
        det_arrays = np.vstack(det_arrays)
        avg_bbox_array = np.sum(scores[order_vote][:, np.mewaxis] * det_arrays, axis=0) / vote_score_sum 
        bbox = BBox.array2bbox(avg_bbox_array)
        bbox.s = scores[index]
        result.append(bbox)
        result_types.append(inst_types[index])
    else:
        result.append(dets[index])
        result_types.append(inst_types[index])
        
    # delete the overlapped bboxes 
    delete_idxes = related_idxes[related_inds]
    in_mask = np.isin(order, delete_idxes, invert=True)
    
    return result, result_types  


        
# nuscenes Loader 

def transform_matrix(translation=np.array([0,0,0]), rotation=np.array([1, 0,0,0]), inverse=False):
    tm = np.eye(4)
    rotation = Quaternion(rotation)
    
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv 
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix 
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm 

def nu_array2mot_bbox(b):
    nu_box = Box(b[:3], b[3:6], Quaternion(b[6:10]))
    mot_bbox = BBox(x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2], w=nu_box.wlh[0],
                    l=nu_box.wlh[1], h=nu_box.wlh[2], o=nu_box.orientation.yaw_pitch_roll[0])
    
    if len(b) == 11:
        mot_bbox.s = b[-1]
    return mot_bbox 



def velo2world(ego_matrix, velo):
    
    """ transform local velocity [x, y] to global
    """
    
    new_velo = velo[:, np.newaxis]
    new_velo = ego_matrix[:2, :2] @ new_velo
    return new_velo[:, 0]



# nuscene data loader

class NuScenesLoader:
    def __init__(self, configs, type_token, segment_name, data_folder, det_data_folder, start_frame):
        """ 
        Initialize with the path to the data.
        
        : param: data_folder(str): root path to your data.
        """
        
        self.configs = configs 
        self.segment = segment_name 
        self.data_loader = data_folder 
        self.det_data_folder = det_data_folder 
        self.type_token = type_token 
        
        self.ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(segment_name)),'r'))
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), allow_pickle=True)
        self.dets = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)), allow_pickle=True)
        self.det_type_filter = True   
        
        self.nms = configs['data_loader']['nms']
        self.nms_thres = configs['data_loader']['nms_thres']
        
        self.max_frame = len(self.dets['bboxes'])
        self.cur_frame = start_frame 
        
        
    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration 
        
        result = dict() 
        ego = self.ego_info[str(self.cur_frame)]
        ego_matrix = transform_matrix(ego[:3], ego[3:])
        result['ego'] = ego_matrix 
    
        bboxes = self.dets['bboxes'][self.cur_frame]
        inst_types = self.dets['types'][self.cur_frame]
        frame_bboxes = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['det_types'] = [inst_types[i] for i in range(len(inst_types)) if inst_types[i] in  self.type_token]
        result['dets'] = [nu_array2mot_bbox(b) for b in frame_bboxes]
        result['aux_info'] = dict()
        if 'velos' in list(self.dets.keys()):
            cur_velos = self.dets['velos'] = [cur_velos[i] for i in rnge(len(cur_velos)) if inst_types[i] in self.type_token]
        else:
            result['aux_info']['velos'] = None 
        
        if self.nms:
             result['dets'], result['det_types'], result['aux_info']['velos'] = self.frame_nms(result['dets'], result['det_types'], resutl['aux_info']['velos'], self.nms_thres)
    
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]
    
        result['pc'] = None 
        
        if 'velos' in list(self.dets.keys()):
            cur_frame_velos = self.dets['velos'][self.cur_frame]
            result['aux_info']['velos'] = [cur_frame_velos[i] for i in  range(len(bboxes)) if inst_types[i] in self.type_token]
        
        result['time_stamp'] = self.ts_info[self.cur_frame] * 1e-6
        result['aux_info']['is_key_frame'] = True  
        
        #result['time_stamp'] = self.ts_info[self.cur_frame][0] * 1e-6 
        #result['aux_info']['is_key_frame'] = self.ts_info[self.cur_frame][1]
        
        self.cur_frame += 1
        return result 
    
    
    def __len__(self):
        return self.max_frame 
    
    
    def frame_nms(self, dets, det_types, velos, thres):
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None 
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
            
        return result_dets, frame_types, result_velos 
    
class NuScenesLoader10Hz:
    def __init__(self, configs, type_token, segment_name, data_folder, det_data_folder, start_frame):
        """
        Initialize with the path to data
        :param data_folder(str): root path to your data
        """
        
        self.configs = configs 
        self.segment = segment_name 
        self.data_loader = data_folder 
        self.det_data_folder = det_data_folder 
        self.type_token = type_token 
        
        
        self.ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(segment_name)), 'r'))
        self.time_stamps = [t[0] for t in self.ts_info]
        self.is_key_frames = [t[1] for t in self.ts_info]
        
        self.token_info = json.load(open(os.path.join(data_folder, 'token_info', '{:}.json'.format(segment_name)), 'r'))
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segmetn_name)), allow_pickle=True)
        self.det_type_filter =  True  
        
        self.max_frame = len(self.dets['bboxes'])
        self.selected_frames = [i for i in range(self.max_frame) if self.token_info[i][3]]
        self.cur_selected_index = 0
        self.cur_frame = start_frame 
        self.nms_thres = configs['running']['nms_thres']
        #import pdb 
        # pdb.set_trace()
        
        
    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.cur_selected_index >= len(self.selected_frames):
            raise StopIteration 
        
        self.cur_frame = self.selected_frames[self.cur_selected_index]
        
        
        result = dict() 
        result['time_stamp'] = self.time_stamps[self.cur_frame] * 1e-6
        ego = self.ego_info[str(self.cur_frame)]
        ego_matrix = transform_matrix(ego[:3], ego[3:])
        result['ego'] = ego_matrix 
        
        bboxes = self.dets['bboxes'][self.cur_frame]
        inst_types = self.dets['types'][self.cur_frame]
        frame_bboxes = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in  self.type_token]
        result['det_types'] = [inst_types[i] for i in range(len(inst_types)) if inst_types[i] in self.type_token]
        
        #result['dets] =[BBox.bbox2array(nu_aray2mot_bbox(b)) for b in frame_bboxes]
        result['dets'] = [nu_array2mot_bbox(b) for b in frame_bboxes]
        
        result['aux_info'] = dict()
        if 'velos' in list(self.dets.keys()):
            cur_velos = self.dets['velos'][self.cur_frame]
            result['aux_info']['velos'] = [cur_velos[i] for i in range(len(cur_velos)) if inst_type[i] in self.type_token]
        else:
            result['aux_info']['velos'] = None 
            
        if self.nms:
            result['dets'], result['det_types'], resukt['aux_info']['velos'] = self.frame_nms(result['dets'], resutl['det_types'], resutl['aux_info']['velos'], self.nms_thres)
            
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]
        
        result['pc'] = None
        
        result['aux_info']['is_key_frame'] = self.is_key_frames[self.cur_frame]
        
        self.cur_seleted_index += 1
        
        return result 
    
    def __len__(self):
        return len(self.selected_frames)
    
    def frame_nms(self, dets, det_types, velos, thres):
        frame_indexes, frame_types = nms(dets, det_types, thres) 
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None 
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos
    
     
     
# wamoLoader.py

class WaymoLoader:
    def __init__(self, configs, type_token, segment_name, data_folder, det_data_folder, start_frame):
        """ Initialize with the path to data.
        : param data_folder(str): root path to your data.
        """
        
        self.configs = configs
        self.segment = segment_name 
        self.data_loader = data_folder 
        self.det_data_folder = det_data_folder 
        self.type_token = type_token 
        
        self.nms = configs['data_loader']['nms']
        self.nms_thres = configs['data_loader']['nms_thres']
        
        self.ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(segment_name)), 'r'))
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), allow_pickle=True)
        
        # vehicle dets
        self.det_type_filter =  True   
        
        self.max_frame = len(self.dets['bboxes'])
        self.cur_frame = start_frame 
        
    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration 
        
        result = dict() 
        result['time_stamp'] = self.ts_info[self.cur_frame] * 1e-6 
        result['ego'] = self.ego_info[str(self.cur_frame)]
        
        bboxes = self.dets['bboxes'][self.cur_frame]
        inst_types = self.dets['types'][self.cur_frame]
        selected_dets = [ bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['det_types'] = [inst_types[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['dets'] = [BBox.bbox2world(result['ego'], BBox.array2bbox(b)) for b in selected_dets]
        
        
        result['pc'] = None 
        result['aux_info'] = {'is_key_frame': True}
        
        
        if 'velos' in self.dets.keys():
            cur_frame_velos = self.dets['velos'][self.cur_frame]
            result['aux_info']['velos'] = [ np.array(cur_frame_velos[i]) for i in range(len(bboxes)) if inst_types[i] in self.type_token]
            result['aux_info']['velos'] = [velo2world(result['ego'], v) for v in result['aux_info']['velos']]
            
        else:
            result['aux_info']['velos'] = None 
            
        if self.nms:
            result['dets'], result['det_types'], result['aux_info']['velos'] = self.frame_nms(result['dets'], result['det_types'], result['aux_info']['velos'], self.nms_thres)
            
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]
        
        self.cur_frame += 1
        
        return result 
    
    
    def __len__(self):
        return self.max_frame 
    
    def frame_nms(self, dets, det_types, velos,thres):
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None 
        if velos is not None:
            resutl_velos = [velos[i] for i in  frame_indexes]
        return result_dets, frame_types, result_velos 
    
    
        
        
        
        
    
    
        
    
    
       
        
        
        


                   
         






