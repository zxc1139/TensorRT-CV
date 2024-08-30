# importing required libraries 
import cv2 
import time 
import numpy as np
import argparse
import torch
import time
import os
from threading import Thread
import multiprocessing
from yolo_triton_inference import YoloTritonInference 
from depth_trion_inference import DepthTritonInference
from utils import get_triton_client, concatenate_imgs, concatenate_4_imgs, concatenate_6_imgs
from stream_source import VideoStream

def stream_loop(queue_from_stream, src):
    cap = cv2.VideoCapture(src)
    # counter = 0
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            # counter += 1
            frame = cv2.resize(frame, (480, 300))
            if len(str(src)) > 5:
                win_name = 'raw_frame_' + str(src[-5:])
            else:
                win_name = 'raw_frame_cam' + str(src)
            frame_dict = {}
            frame_dict['source'] = src
            frame_dict['frame_data'] = frame
            frame_dict['frame_num'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # frame_dict['frame_num'] = counter
            frame_dict['frame_window'] = win_name
            queue_from_stream.put(frame_dict)
            # cv2.imshow(win_name, frame)
            if ret is False :
                print('[Exiting] No more frames to read')
                break 


def main(video_path, batch, url, mode):
    triton_client = get_triton_client(url)

    # Depth-Anything wrapper
    depth_wrapper = DepthTritonInference(triton_client=triton_client, 
                                         model_name='depth-anything-vits14', 
                                         model_version='1',
                                         image_path=None,
                                         mode=mode)

    # Yolov8 wrapper
    yolov8s_wrapper = YoloTritonInference(triton_client=triton_client,
                                          model_name='yolov8s',
                                          model_version='1', #5:dynamic-int8, 4:dynamic-fp16
                                          image_path=None,
                                          mode=mode)
    
    yolov8n_wrapper = YoloTritonInference(triton_client=triton_client,
                                          model_name='yolov8n',
                                          model_version='1', #3:dynamic-int8, 2:dynamic-fp16
                                          image_path=None,
                                          mode=mode)
    

    print('initializing stream')
    queue_from_stream = multiprocessing.Queue()
    stream_process = multiprocessing.Process(target=stream_loop, args=(queue_from_stream, 'rtsp://10.20.0.35:8554/ws7-a'))
    stream_process.start()
    stream_process2 = multiprocessing.Process(target=stream_loop, args=(queue_from_stream, 'rtsp://10.20.0.35:8554/ws7-b'))
    stream_process2.start()
    stream_process3 = multiprocessing.Process(target=stream_loop, args=(queue_from_stream, 'rtsp://10.20.0.35:8554/ws7-c'))
    stream_process3.start()
    stream_process4 = multiprocessing.Process(target=stream_loop, args=(queue_from_stream, 0))
    stream_process4.start()
    batched_img = []
    while True:
        if queue_from_stream.empty():
            continue
        curr_frame = queue_from_stream.get()
        cv2.imshow(curr_frame['frame_window'], curr_frame['frame_data'])
        print(curr_frame['frame_window'], curr_frame['frame_num'])

        # depth_wrapper.read_image(raw_frame=curr_frame['frame_data'])
        # depth_wrapper.run_inference()
        # out_depth = depth_wrapper.save_output()
        # depth_window = 'depth_' + curr_frame['frame_window'][-5:]
        # cv2.imshow(depth_window, out_depth)

        # start = time.time()
        # yolov8s_wrapper.read_image(raw_frame=curr_frame['frame_data'])
        # yolov8s_wrapper.run_inference()
        # yolo_output = yolov8s_wrapper.save_output()
        # yolo_window = 'yolo_' + curr_frame['frame_window'][-5:]
        # cv2.imshow(yolo_window, yolo_output)

        
        # if len(batched_img) < 6:
        #     batched_img.append(curr_frame)
        # if len(batched_img) == 6:
     
        #     '''depth-anything batched images inference'''
        #     start = time.time()
        #     depth_wrapper.read_image_stack(folder_dir=None, 
        #                                     mode='video', 
        #                                     img_list=batched_img) 
        #     batched_out_d1, infer_fps = depth_wrapper.send_async_requests(batch_size=batch) 
        #     print('depth inference throughput ', infer_fps)
        #     end = time.time()
        #     totalTime = end - start
        #     fps = 1 / totalTime
            
        #     for i in range(len(batched_img)):
        #         # cv2.putText(res_img, f'Batched Inference Throughput: {int(infer_fps1)}', 
        #         #     (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        #         new_window = 'depth_' + batched_img[i]['frame_window'][-5:]
        #         cv2.imshow(new_window, batched_out_d1[i])
        #     print("Depth-Anything images processing throughput: ", fps)


        #     '''Yolov8s batched images inference'''
        #     start = time.time()
        #     yolov8s_wrapper.read_image_stack(folder_dir=None, 
        #                                         mode='video', 
        #                                         img_list=batched_img) 
        #     batched_out_o1, infer_fps = yolov8s_wrapper.send_async_requests(batch_size=batch)
        #     print('yolo inference throughput ', infer_fps)
        #     end = time.time()
        #     totalTime = end - start
        #     fps = 1 / totalTime
        #     for i in range(len(batched_img)):
        #         new_window = 'yolo_' + batched_img[i]['frame_window'][-5:]
        #         cv2.imshow(new_window, batched_out_o1[i])
        #     print("Yolov8s images processing throughput: ", fps)
        #     batched_img = []

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    print("Destroying process...")
    stream_process.terminate()
    stream_process2.terminate()
    stream_process3.terminate()
    stream_process4.terminate()

    stream_process.join()
    stream_process2.join()
    stream_process3.join()
    stream_process4.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None) #'./data/davis_dolphins.mp4'
    parser.add_argument('--batch', type=int, default=2) #batch number of requests
    parser.add_argument('--url', type=str, default='localhost:8001')
    parser.add_argument('--mode', type=str, default='video')
    args = parser.parse_args()
    main(args.video_path, args.batch, args.url, args.mode)