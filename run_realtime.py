import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
from dv import AedatFile
from dv import NetworkEventInput
import threading
import queue

q=queue.Queue()

class camerastream(threading.Thread):
    def __init__(self,q):
        super(camerastream,self).__init__()
        self.dataline=np.empty([1,4])
        self.stopped=False
        self.q =q
    

    def run(self):
        self.dataline=np.empty([1,4])
        while True:
            with NetworkEventInput(address='127.0.0.1', port=7777) as i:
                for event in i:
                    timestamps = np.float64(event.timestamp)
                    x = np.int16(event.x) 
                    y = np.int16(event.y) 
                    polarities = np.int16(event.polarity)
                    tmp=np.array([[timestamps, x, y, polarities]])
                    self.dataline = np.concatenate((self.dataline, tmp), axis=0)
                    if len(self.dataline) == 15000 :
                        #print(self.dataline)
                        self.q.put(self.dataline)
                        
                        self.dataline=np.empty([1,4])



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', type=str,default= 'pretrained/E2VID_lightweight.pth.tar',
                        help='path to model weights')
    #parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('-r', '--row_file', type = str)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    width=346
    height=260
    print('Sensor size: {} x {}'.format(width, height))

    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    """ Read chunks of events using Pandas """
    '''
    # Loop through the events and reconstruct images
    N = None
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
    '''
    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    thread1= camerastream(q)
    thread1.start( )
    with Timer('Processing entire dataset'):
        while True:
            if q.empty():
                event_window=q.get()
                last_timestamp = event_window[-1, 0]

                with Timer('Building event tensor'):
                    if args.compute_voxel_grid_on_cpu:
                        event_tensor = events_to_voxel_grid(event_window,
                                                            num_bins=model.num_bins,
                                                            width=width,
                                                            height=height)
                        event_tensor = torch.from_numpy(event_tensor)
                    else:
                        event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                    num_bins=model.num_bins,
                                                                    width=width,
                                                                    height=height,
                                                                    device=device)

                num_events_in_window = event_window.shape[0]
                reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

                start_index += num_events_in_window


'''

    with Timer('Processing entire dataset'):
        for event_window in event_window_iterator:
            #print(event_window)
            last_timestamp = event_window[-1, 0]

            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=model.num_bins,
                                                        width=width,
                                                        height=height)
                    event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)

            num_events_in_window = event_window.shape[0]
            reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

            start_index += num_events_in_window

'''