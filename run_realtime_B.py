from PyQt5 import QtWidgets, QtGui, QtCore
from Ui_Hello import Ui_MainWindow
import torch
import sys
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
import datetime
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
from dv import AedatFile
from dv import NetworkNumpyEventPacketInput
import threading
import queue
from models.experimental import attempt_load
con = threading.Condition()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,args):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label.setFont(QtGui.QFont('Arial', 10))
        self.ui.label.setText('Nothing')
        self.ui.lineEdit.setText('Welcome!')

        self.ui.pushButton.setText('Display')
        self.ui.pushButton.clicked.connect(self.buttonClicked)

        self.ui.horizontalScrollBar.setMinimum(0)
        self.ui.horizontalScrollBar.setMaximum(100)
        self.ui.horizontalScrollBar.setValue(0)
        self.ui.horizontalScrollBar.valueChanged.connect(self.sliderValue)

        self.ui.imgbutton.clicked.connect(self.graphicView)

    
        height=260;width=346
        print('Sensor size: {} x {}'.format(width, height))

        # Load model
        model = load_model(args.path_to_model)
        device = get_device(args.use_gpu)
        model = model.to(device)
        model.eval()
        model_yolo = attempt_load(['yolov5m.pt'], map_location=device)
        model_yolo.eval()
        reconstructor = ImageReconstructor(model, height, width, model.num_bins, args,model_yolo)
        if args.compute_voxel_grid_on_cpu:
            print('Will compute voxel grid on CPU.')

        self.thread1= camerastream(args = args,device=device,reconstructor=reconstructor,model=model)
        self.thread2 = Thread2()

        #thread1.setDaemon(True)
        self.thread2.start()
        self.thread1.start() 
        
    def sliderValue(self):
        value = self.ui.horizontalScrollBar.value()
        self.ui.sclabel.setText(str(value))

    def buttonClicked(self):
        text = self.ui.lineEdit.text()
        self.ui.label.setText(text)
        self.ui.lineEdit.clear()

    def graphicView(self,x):
        img = x
        img = QtGui.QPixmap(img).scaled(self.ui.imglabel.width(), self.ui.imglabel.height())
        self.ui.imglabel.setPixmap(x)

        
class camerastream(threading.Thread):
    def __init__(self,args,device,reconstructor,model):
        super(camerastream,self).__init__()
        self.dataline=np.empty([1,4])
        self.args = args
        self.width = 346
        self.height = 260
        self.reconstructor = reconstructor
        self.device = device
        self.model = model
        initial_offset = args.skipevents
        sub_offset = args.suboffset
        self.start_index = initial_offset + sub_offset
    
    def run(self):
        while True:
            con.acquire(timeout=1e-4)
            #lock.acquire()
            with Timer('Processing entire dataset'):
                con.wait()
                event_window=q.get()
                con.release()
                last_timestamp = event_window[-1, 0]
                with Timer('Building event tensor'):    
                    if args.compute_voxel_grid_on_cpu:
                        event_tensor = events_to_voxel_grid(event_window,
                                                            num_bins=self.model.num_bins,
                                                            width=self.width,
                                                            height=self.height)
                        event_tensor = torch.from_numpy(event_tensor)
                    else:
                        event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                    num_bins=self.model.num_bins,
                                                                    width=self.width,
                                                                    height=self.height,
                                                                    device=self.device)
                num_events_in_window = event_window.shape[0]
                self.reconstructor.update_reconstruction(event_tensor, self.start_index+ num_events_in_window, last_timestamp)
                self.start_index += num_events_in_window
        


class Thread2(threading.Thread):
    def __init__(self):
        super(Thread2,self).__init__()

    def run(self): 
        dataline=np.empty([1,4])
        time_start = time.time()
        time_stop = time.time()
        time_tmp = 0
        #while True:
        with NetworkNumpyEventPacketInput(address='127.0.0.1', port=7777) as i:
                time_start = time.time()
                for event in i:
                    timestamps = event['timestamp']
                    x = event['x']
                    y = event['y']
                    polarities = event['polarity']
                    tmp=np.array([[timestamps, x, y, polarities]])
                    tmp = tmp.T.reshape(-1,4)
                    dataline = np.concatenate((dataline, tmp), axis=0)
                    if len(dataline) >= 15000 :
                        con.acquire(timeout=1e-4)
                        q.put(dataline)
                        time_stop = time.time()
                        print('Total Time:'+str(time_stop - time_start-time_tmp))
                        time_tmp = time_stop - time_start   
                        dataline=np.empty([1,4])
                        con.notifyAll()
                        #import pdb;pdb.set_trace()
                        con.release()





if __name__ == "__main__":
    global q
    q=queue.Queue()
    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', type=str,default= 'pretrained/firenet_1000.pth.tar',
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

    

    app = QtWidgets.QApplication([])
    window = MainWindow(args)
    window.show()
    app.exec_()





        


    
    
