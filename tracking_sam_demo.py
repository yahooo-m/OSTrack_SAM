import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
from collections import OrderedDict
import importlib
from lib.test.evaluation.environment import env_settings
import time
from pathlib import Path


from segment_anything import build_sam, SamPredictor 
import numpy as np
import matplotlib.pyplot as plt

from lib.models.ostrack import build_ostrack
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset, run_sequence
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
from lib.train.data.processing_utils import transform_image_to_crop
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.lmdb_utils import decode_img
from lib.utils.box_ops import box_xywh_to_xyxy


## build SAM 
def build_SAM(checkpoint):
    predictor = SamPredictor(build_sam(checkpoint=checkpoint))
    return predictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2]-box[0], box[3]-box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

class TrackerSAM:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, checkpoint: int = None,display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        self.predictor = build_SAM(checkpoint=checkpoint)

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              'lib/test', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            self.predictor.set_image(image)
            tgt_box = torch.tensor(out['target_bbox'])
            tgt_box_xyxy = box_xywh_to_xyxy(tgt_box)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(tgt_box_xyxy, image.shape[:2])
            
            masks, _, _ = self.predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
                )

            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(masks.cpu().numpy(), plt.gca(), random_color=False)
            show_box(tgt_box_xyxy.numpy(), plt.gca())
            plt.axis('off')
            os.makedirs(os.path.join('/home/deshui/pro/OSTrack/output/result', frame_path.split('/')[-2]), exist_ok=True)
            plt.savefig(os.path.join('/home/deshui/pro/OSTrack/output/result', frame_path.split('/')[-2], "{}.jpg".format(frame_num)), bbox_inches="tight")
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, checkpoint=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = dataset[sequence]

    
    if sequence is None:
        trackers = [TrackerSAM(tracker_name, tracker_param, dataset_name, run_id, checkpoint=checkpoint)]
        run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)
    else:
        trackers = TrackerSAM(tracker_name, tracker_param, dataset_name, run_id, checkpoint=checkpoint)
        run_sequence(seq=dataset,tracker= trackers,debug= debug, num_gpu=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, required=True, help='sam checkpoint')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()



'''
### load OSTrack
def get_track(track_params, checkpoint):
    network = build_ostrack(track_params, training=False)
    network.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    return network

### build tracker


### load data

def get_dataset(dataset_name, sequence=None):
    dataset = get_dataset(dataset_name)
    if sequence is not None:
        dataset = [dataset[sequence]]
    return dataset

### processing dataset


### read image
def read_image(self, image_file: str):
    if isinstance(image_file, str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    elif isinstance(image_file, list) and len(image_file) == 2:
        return decode_img(image_file[0], image_file[1])
    else:
        raise ValueError("type of image_file should be str or list")


### transform template box

def transform_bbox_to_crop(box_in, resize_factor, device, box_extract=None, crop_type='template'):
    # box_in: list [x1, y1, w, h], not normalized
    # box_extract: same as box_in
    # out bbox: Torch.tensor [1, 1, 4], x1y1wh, normalized
    if crop_type == 'template':
        crop_sz = torch.Tensor([192, 192])
    elif crop_type == 'search':
        crop_sz = torch.Tensor([384, 384])
    else:
        raise NotImplementedError

    box_in = torch.tensor(box_in)
    if box_extract is None:
        box_extract = box_in
    else:
        box_extract = torch.tensor(box_extract)
    template_bbox = transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=True)
    template_bbox = template_bbox.view(1, 1, 4).to(device)

    return template_bbox

### process template
def initialize(image, info: dict, template_factor, template_size, ce_loc, cfg, save_all_boxes=True):
    # forward the template once
    z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], template_factor,
                                                output_sz=template_size)
    z_patch_arr = z_patch_arr
    preprocessor = Preprocessor()
    template = preprocessor.process(z_patch_arr, z_amask_arr)
    with torch.no_grad():
        z_dict1 = template

    box_mask_z = None
    if ce_loc:
        template_bbox = transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                    template.tensors.device).squeeze(1)
        box_mask_z = generate_mask_cond(cfg, 1, template.tensors.device, template_bbox)

    # save states
    state = info['init_bbox']
    frame_id = 0
    if save_all_boxes:
        all_boxes_save = info['init_bbox'] * cfg.MODEL.NUM_OBJECT_QUERIES
        return {"all_boxes": all_boxes_save}

def track_sequence(tracker, seq, init_info):
    # Define outputs
    # Each field in output is a list containing tracker prediction for each frame.

    # In case of single object tracking mode:
    # target_bbox[i] is the predicted bounding box for frame i
    # time[i] is the processing time for frame i

    # In case of multi object tracking mode:
    # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
    # frame i
    # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
    # object in frame i

    output = {'target_bbox': [],
                'time': []}
    if tracker.params.save_all_boxes:
        output['all_boxes'] = []
        output['all_scores'] = []

    def _store_outputs(tracker_out: dict, defaults=None):
        defaults = {} if defaults is None else defaults
        for key in output.keys():
            val = tracker_out.get(key, defaults.get(key, None))
            if key in tracker_out or val is not None:
                output[key].append(val)

    # Initialize
    image = read_image(seq.frames[0])

    out = initialize(image, init_info)
    if out is None:
        out = {}

    prev_output = OrderedDict(out)
    init_default = {'target_bbox': init_info.get('init_bbox'),}
    if tracker.params.save_all_boxes:
        init_default['all_boxes'] = out['all_boxes']
        init_default['all_scores'] = out['all_scores']

    _store_outputs(out, init_default)

    for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
        image = read_image(frame_path)


        info = seq.frame_info(frame_num)
        info['previous_output'] = prev_output

        if len(seq.ground_truth_rect) > 1:
            info['gt_bbox'] = seq.ground_truth_rect[frame_num]
        out = tracker.track(image, info)
        prev_output = OrderedDict(out)
        _store_outputs(out)

    for key in ['target_bbox', 'all_boxes', 'all_scores']:
        if key in output and len(output[key]) <= 1:
            output.pop(key)

    return output
'''
