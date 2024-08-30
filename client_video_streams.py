# importing required libraries 
import cv2 
import time 
import numpy as np
import argparse
import torch
import time
import os
from threading import Thread
from yolo_triton_inference import YoloTritonInference 
from depth_trion_inference import DepthTritonInference
from utils import get_triton_client, concatenate_imgs, concatenate_4_imgs, concatenate_6_imgs
from stream_source import VideoStream


def main(video_path, batch, url, mode):
    triton_client = get_triton_client(url)

    # Depth-Anything wrapper
    depth_wrapper = DepthTritonInference(triton_client=triton_client, 
                                         model_name='depth-anything-vits14', 
                                         model_version='1',
                                         image_path=None,
                                         mode=mode)

    # Yolov8s wrapper
    yolov8s_wrapper = YoloTritonInference(triton_client=triton_client,
                                          model_name='yolov8s',
                                          model_version='1', 
                                          image_path=None,
                                          mode=mode)
    
    yolov8n_wrapper = YoloTritonInference(triton_client=triton_client,
                                          model_name='yolov8n',
                                          model_version='1', 
                                          image_path=None,
                                          mode=mode)
        

    video_stream_widget_1 = VideoStream(src=0)
    video_stream_widget_1.start()

    video_stream_widget_2 = VideoStream(src='rtsp://10.20.0.35:8554/ws7-a') 
    video_stream_widget_2.start()

    video_stream_widget_3 = VideoStream(src='rtsp://10.20.0.35:8554/ws7-b') 
    video_stream_widget_3.start()

    video_stream_widget_4 = VideoStream(src='rtsp://10.20.0.35:8554/ws7-c')
    video_stream_widget_4.start()

    video_stream_widget_5 = VideoStream(src='rtsp://10.20.0.35:8554/ws7-b')
    video_stream_widget_5.start()

    video_stream_widget_6 = VideoStream(src='rtsp://10.20.0.35:8554/ws7-c') 
    video_stream_widget_6.start()

    # counter = 0
    batched_l = []
    while True: 
        try:
            batched_images = []

            # frame_to_process = {
            #     content: nparrayt,
            #     id: 1,
            #     streamname:
            #     inference_result: asd
            # }
            #frames_to_process = JessiesFrameQueue.getNext(20)

            _, frame_num1, frame1 = video_stream_widget_1.get_frame()
            _, frame_num2, frame2 = video_stream_widget_2.get_frame()
            _, frame_num3, frame3 = video_stream_widget_3.get_frame()
            _, _, frame4 = video_stream_widget_4.get_frame()
            _, _, frame5 = video_stream_widget_5.get_frame()
            _, _, frame6 = video_stream_widget_6.get_frame()
            
            print(frame_num2, frame_num3)
            # counter += 1

            raw_conh = concatenate_6_imgs(frame1, frame2, frame3, frame4, frame5, frame6)
            cv2.imshow('raw_concat', raw_conh)

            batched_images = [frame1, frame2, frame3, frame4, frame5, frame6]
            # batched_l.extend(batched_images)

            # if counter % 2 == 0: # every 2 frames
            #     '''depth-anything batched images inference'''
            #     start = time.time()
            #     depth_wrapper.read_image_stack(folder_dir=None, 
            #                                     mode='video', 
            #                                     img_list=batched_l) 
            #     batched_out_d1, infer_fps1 = depth_wrapper.send_async_requests(batch_size=batch) 
            #     # print(len(batched_out_d1))
            #     print('depth throughput ', infer_fps1)
            #     end = time.time()
            #     totalTime = end - start
            #     fps = 1 / totalTime
            #     print("Depth-Anything whole batched images processing throughput: ", fps)     

            #     '''Yolov8s batched images inference'''
            #     start = time.time()
            #     yolov8s_wrapper.read_image_stack(folder_dir=None, 
            #                                         mode='video', 
            #                                         img_list=batched_l) 
            #     batched_out_o1, infer_fps1 = yolov8s_wrapper.send_async_requests(batch_size=batch)
            #     print(len(batched_out_o1))
            #     print('object detection throughput ', infer_fps1)
            #     end = time.time()
            #     totalTime = end - start
            #     fps = 1 / totalTime
            #     print("Yolo whole batched images processing throughput: ", fps)
                
            #     batched_l = []


            '''depth-anything batched images inference'''
            start = time.time()
            depth_wrapper.read_image_stack(folder_dir=None, 
                                            mode='video', 
                                            img_list=batched_images) 
            batched_out_d1, infer_fps1 = depth_wrapper.send_async_requests(batch_size=batch) 
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(batched_out_d1[0], f'Batched Inference Throughput: {int(infer_fps1)}', (20,70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            depth_con = concatenate_6_imgs(batched_out_d1[0], batched_out_d1[1],
                                           batched_out_d1[2], batched_out_d1[3],
                                           batched_out_d1[4], batched_out_d1[5])
            cv2.imshow('depth_concat', depth_con)
            print("Depth-Anything batched images processing throughput: ", fps)     


            
            '''Yolov8s batched images inference'''
            start = time.time()
            yolov8s_wrapper.read_image_stack(folder_dir=None, 
                                                mode='video', 
                                                img_list=batched_images) 
            batched_out_o1, infer_fps1 = yolov8s_wrapper.send_async_requests(batch_size=batch)
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(batched_out_o1[0], f'Batched Inference Throughput: {int(infer_fps1)}', (20,70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # yolov8s_con = concatenate_4_imgs(batched_out_o1[0], batched_out_o1[1],
            #                                   batched_out_o1[2], batched_out_o1[3])
            yolov8s_con = concatenate_6_imgs(batched_out_o1[0], batched_out_o1[1],
                                                batched_out_o1[2], batched_out_o1[3],
                                                batched_out_o1[4], batched_out_o1[5])
            cv2.imshow('yolov8s_concat', yolov8s_con)
            print("Yolov8s images batched images processing throughput: ", fps)


            # '''depth-anything single image inference'''
            # start = time.time()
            # depth_wrapper.read_image(raw_frame=frame1)
            # depth_wrapper.run_inference()
            # out_depth1 = depth_wrapper.save_output()
            
            # '''Yolov8s single image inference'''
            # start = time.time()
            # yolov8s_wrapper.read_image(raw_frame=frame1)
            # yolov8s_wrapper.run_inference()
            # out_yolov8s1 = yolov8s_wrapper.save_output()

        except AttributeError:
            pass

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None) #'./data/davis_dolphins.mp4'
    parser.add_argument('--batch', type=int, default=2) #batch number of requests
    parser.add_argument('--url', type=str, default='localhost:8001')
    parser.add_argument('--mode', type=str, default='video')
    args = parser.parse_args()
    main(args.video_path, args.batch, args.url, args.mode)
