import logging
import time
import datetime
from multiprocessing import Queue, Pool
from queue import Empty
from threading import Thread
import numpy as np
import cv2
import json
import skvideo.io
from yolo_triton_inference import YoloTritonInference 
from depth_trion_inference import DepthTritonInference
from utils import get_triton_client, concatenate_6_imgs, concatenate_imgs

class FrameData():
    def __init__(self, frame, frame_num, manual_num):
        self.frame = frame
        self.frame_num = frame_num
        self.manual_num = manual_num
        # self.frame_time = time.time()
        self.frame_time = datetime.datetime.now()

class VideoStream(object):
    def __init__(self, src: str, models: list):
        self.capture = cv2.VideoCapture(src)
        # self.capture = skvideo.io.vread(src)
        self.thread = Thread(target=self.update, args=())
        self.src = src
        self.thread.daemon = True
        self.stopped = True
        self.models = models #list
        self.err_message = None
        self.frame_data = None

        # frame_class
        # timespame
        # frame number
        # frame


    def start(self):
        self.stopped = False
        self.thread.start()

    def stop(self):
        self.stopped = True 
        self.thread.join()
        # self.capture.release()
        cv2.destroyAllWindows()
        # exit(1)

    def update(self):
        # Read the next frame from the stream in a different thread
        frames_count = 0
        while True:
            if self.stopped is True:
                break
           
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                frame_num = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                frames_count += 1
                self.frame_data = FrameData(self.frame, frame_num, frames_count)
                if self.status is False :
                    print('[Exiting] No more frames to read')
                    self.stopped = True
                    break 
            else:
                self.err_message = 'Cannot start video stream'
            # time.sleep(.005)


    def get_frame(self):
        frame_num = self.frame_data.frame_num
        manual_num = self.frame_data.manual_num
        frame = self.frame_data.frame
        frame_time = self.frame_data.frame_time
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
        return frame_num, manual_num, frame, frame_time
    
    def get_model(self):
        return self.models
    
    def update_model(self, update_model):
        self.models = update_model
        return self.models
    
    def get_err_message(self):
        return self.err_message