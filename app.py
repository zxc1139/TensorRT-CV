import cv2
import os
import time
from threading import Thread
from flask import Flask, request, jsonify
from stream_source import VideoStream
from utils import get_triton_client, concatenate_6_imgs, concatenate_imgs
from yolo_triton_inference import YoloTritonInference 
from depth_trion_inference import DepthTritonInference
import matplotlib.pyplot as plt
from PIL import Image
import queue
import logging
import datetime
from csv_logger import CsvLogger

app = Flask(__name__, instance_relative_config=True)
count = 0
yolov8s_counter = 0
yolov8n_counter = 0
depth_counter = 0
NUM_STREAMS = os.getenv("NUM_STREAMS")
BATCH_SIZE = os.getenv("BATCH_SIZE")
TRITON_URL = os.getenv("TRITON_URL")
MODE = os.getenv("MODE")

filename = 'logs/videostream_log.csv'
delimiter = ','
level = logging.INFO
# custom_additional_levels = ['capture', 'inference', 'result']
fmt = fmt = f'%(asctime)s{delimiter}%(message)s'
datefmt = '%Y/%m/%d %H:%M:%S'

header = ['date', 'model_type', 'video_source', 'frame_number', 'capture_time','infer_time', 'get_result_time']

# Creat logger with csv rotating handler
csvlogger = CsvLogger(filename=filename,
                      delimiter=delimiter,
                      level=level,
                    #   add_level_names=custom_additional_levels,
                      add_level_nums=None,
                      fmt=fmt,
                      datefmt=datefmt,
                      header=header)



triton_client = get_triton_client(TRITON_URL)
active_streams_dictionary = {}
model_list = [] # a list to store models used for each stream
yolo_results_queue = queue.Queue()
depth_results_queue = queue.Queue()


def _run_single_inference(model_wrapper, batched_images, scr_names, model_type, results_list):
    '''single image inference'''
    model_wrapper.read_image(raw_frame=batched_images[0])
    model_wrapper.run_inference()
    
    if 'yolo' in model_type:
        out_img, bboxes, scores, labels, classes = model_wrapper.save_output()
        model_win = model_type + '_' + scr_names[0]
        file_name = model_win + '.jpg'
        # cv2.imshow(model_win, out_img)
        # plt.imshow(out_img)
        # plt.show()
        cv2.imwrite(file_name, out_img)
        results_dict = results_list[0]
        results_dict['inference_result'] = out_img
        results_dict['bboxes_lst'] = bboxes
        results_dict['scores_lst'] = scores
        results_dict['labels_lst'] = labels
        results_dict['classes_lst'] = classes
        yolo_results_queue.put(results_dict)


    if 'depth' in model_type:
        out_img, raw_depth = model_wrapper.save_output()
        model_win = model_type + '_' + scr_names[0]
        file_name = model_win + '.jpg'
        # cv2.imshow(model_win, out_img)
        cv2.imwrite(file_name, out_img)
        results_dict = results_list[0]
        results_dict['inference_result'] = out_img
        results_dict['raw_depth'] = raw_depth
        depth_results_queue.put(results_dict)

def _run_batched_inference(model_wrapper, batched_images, scr_names, model_type, results_list):
    '''batched images inference'''
    model_wrapper.read_image_stack(folder_dir=None, 
                                    mode='video', 
                                    img_list=batched_images) 
    if 'yolo' in model_type:
        batched_out, batched_bboxes, batched_scores, batched_labels, batched_classes, infer_fps1 = model_wrapper.send_async_requests(batch_size=int(BATCH_SIZE)) 
        cv2.putText(batched_out[0], f'Batched Inference Throughput: {int(infer_fps1)}', (20,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        for scr_name, out_img, bboxes, scores, labels, classes, results_dict in zip(scr_names, batched_out, batched_bboxes, batched_scores, batched_labels, batched_classes, results_list):
            model_win = model_type + '_' + scr_name
            file_name = model_win + '.jpg'
            # cv2.imshow(model_win, out_img)
            cv2.imwrite(file_name, out_img)

            results_dict['inference_result'] = out_img
            results_dict['bboxes_lst'] = bboxes
            results_dict['scores_lst'] = scores
            results_dict['labels_lst'] = labels
            results_dict['classes_lst'] = classes
            yolo_results_queue.put(results_dict)


    if 'depth' in model_type:
        batched_out, batach_raw_depth, infer_fps1 = model_wrapper.send_async_requests(batch_size=int(BATCH_SIZE)) 
        cv2.putText(batched_out[0], f'Batched Inference Throughput: {int(infer_fps1)}', (20,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        for scr_name, out_img, raw_depth, results_dict in zip(scr_names, batched_out, batach_raw_depth, results_list):
            model_win = model_type + '_' + scr_name
            file_name = model_win + '.jpg'
            # cv2.imshow(model_win, out_img)
            cv2.imwrite(file_name, out_img)

            results_dict['inference_result'] = out_img
            results_dict['raw_depth'] = raw_depth
            depth_results_queue.put(results_dict)


def initialize_depth():
    print('Start initializing depth-anything')
    global depth_counter
    depth_wrapper = DepthTritonInference(triton_client=triton_client, 
                                         model_name='depth-anything-vits14', 
                                         model_version='1',
                                         image_path=None,
                                         mode=MODE)
    
    while True:
        if len(active_streams_dictionary) > 0:
            try:
                batched_images_depth = []
                scr_names = []
                depth_results = []
                for k, v in list(active_streams_dictionary.items()):
                    video_scr = k
                    video_stream_widget = v
                    model_type = video_stream_widget.get_model()
                    frame_num, manual_num, frame, frame_time = video_stream_widget.get_frame()  
                
                    if isinstance(video_scr, int): # src should be an integer is video source is camera
                        scr_name = str(video_scr)
                        frame_num = int(manual_num)
                    else:
                        scr_name = video_scr[-5:]
        

                    win_name = 'raw_frame_' + scr_name
                    file_name = win_name + '.jpg'
                    frame = cv2.resize(frame, (480, 300))
                    # cv2.imshow(win_name, frame)
                    cv2.imwrite(file_name, frame)
                    scr_names.append(scr_name)  

                    if 'depth' in model_type:     
                        depth_counter += 1     
                        depth_result_dict = {}
                        depth_result_dict['model_type'] = 'depth-vits14'
                        depth_result_dict['video_source'] = scr_name
                        depth_result_dict['frame_num'] = frame_num
                        depth_result_dict['frame_time'] = frame_time
                        depth_result_dict['raw_frame'] = frame.copy()
                        depth_results.append(depth_result_dict)
                        batched_images_depth.append(frame.copy())
                        # frame_data = list(depth_result_dict.items())[0:4]
                        # logger.info(f'Depth input frame data : {frame_data}')

                
                # run depth_anything inference
                if len(batched_images_depth) == 1:
                    infer_time = datetime.datetime.now()
                    # logger.info(f'Start running depth inference at : {infer_time}')
                    # logger.info(f'Total frames sending to depth-anything : {depth_counter}')
                    _run_single_inference(depth_wrapper, batched_images_depth, scr_names, 'depth', depth_results)
                    result_time = datetime.datetime.now()
                    csvlogger.info([depth_result_dict['model_type'],
                                      depth_result_dict['video_source'],
                                      depth_result_dict['frame_num'],
                                      depth_result_dict['frame_time'],
                                      infer_time,
                                      result_time])

                elif len(batched_images_depth) > 1:
                    infer_time = datetime.datetime.now()
                    # logger.info(f'Start running depth BATCHED inference at : {infer_time}')
                    # logger.info(f'Total frames sending to depth-anything : {depth_counter}')
                    _run_batched_inference(depth_wrapper, batched_images_depth, scr_names, 'depth', depth_results)
                    result_time = datetime.datetime.now()
                    for i in range(depth_results):
                        model_info = depth_results[i]['model_type']
                        video_info = depth_results[i]['video_source']
                        frame_num_info = depth_results[i]['frame_num']
                        frame_time_info = depth_results[i]['frame_time']
                        csvlogger.info([model_info, video_info, frame_num_info, frame_time_info, infer_time, result_time])
            except AttributeError:
                pass

        key = cv2.waitKey(1)
        if key == ord('q'):
            print('End running depth-anything')
            cv2.destroyAllWindows()
            break
        else:
            pass

def initialize_yolov8s():
    print('Start initializing yolov8s')
    # Yolov8s wrapper
    global yolov8s_counter
    yolov8s_wrapper = YoloTritonInference(triton_client=triton_client,
                                          model_name='yolov8s',
                                          model_version='1', 
                                          image_path=None,
                                          mode=MODE)

    while True:
        if len(active_streams_dictionary) > 0:
            try:
                batched_images_yolov8s = []
                scr_names = []
                yolov8s_results = []
                for k, v in list(active_streams_dictionary.items()):
                    video_scr = k
                    video_stream_widget = v
                    model_type = video_stream_widget.get_model()
                    frame_num, manual_num, frame, frame_time = video_stream_widget.get_frame()  
                
                    if isinstance(video_scr, int): # src should be an integer is video source is camera
                        scr_name = str(video_scr)
                        frame_num = int(manual_num)
                    else:
                        scr_name = video_scr[-5:]

                    win_name = 'raw_frame_' + scr_name
                    file_name = win_name + '.jpg'
                    frame = cv2.resize(frame, (480, 300))
                    # cv2.imshow(win_name, frame)
                    cv2.imwrite(file_name, frame)
                    scr_names.append(scr_name)

                    if 'yolov8s' in model_type:
                        yolov8s_counter += 1
                        yolov8s_result_dict = {}
                        yolov8s_result_dict['model_type'] = 'yolov8s'
                        yolov8s_result_dict['video_source'] = scr_name
                        yolov8s_result_dict['frame_num'] = frame_num
                        yolov8s_result_dict['frame_time'] = frame_time
                        yolov8s_result_dict['raw_frame'] = frame.copy()
                        yolov8s_results.append(yolov8s_result_dict)
                        batched_images_yolov8s.append(frame.copy())

               
                # run yolov8s inference
                if len(batched_images_yolov8s) == 1:
                    infer_time = datetime.datetime.now()
                    _run_single_inference(yolov8s_wrapper, batched_images_yolov8s, scr_names, 'yolov8s', yolov8s_results)
                    result_time = datetime.datetime.now()
                    csvlogger.info([yolov8s_result_dict['model_type'],
                                    yolov8s_result_dict['video_source'],
                                    yolov8s_result_dict['frame_num'],
                                    yolov8s_result_dict['frame_time'],
                                    infer_time,
                                    result_time])
                    
                  
                elif len(batched_images_yolov8s) > 1:
                    infer_time = datetime.datetime.now()
                    _run_batched_inference(yolov8s_wrapper, batched_images_yolov8s, scr_names, 'yolov8s', yolov8s_results)
                    result_time = datetime.datetime.now()
                    for i in range(yolov8s_results):
                        model_info = yolov8s_results[i]['model_type']
                        video_info = yolov8s_results[i]['video_source']
                        frame_num_info = yolov8s_results[i]['frame_num']
                        frame_time_info = yolov8s_results[i]['frame_time']
                        csvlogger.info([model_info, video_info, frame_num_info, frame_time_info, infer_time, result_time])
          
            except AttributeError:
                pass

        key = cv2.waitKey(1)
        if key == ord('q'):
            print('End running Yolov8s')
            cv2.destroyAllWindows()
            break
        else:
            pass


def initialize_yolov8n():
    print('Start initializing yolov8n')
    global yolov8n_counter
    # Yolov8s wrapper
    yolov8n_wrapper = YoloTritonInference(triton_client=triton_client,
                                          model_name='yolov8n',
                                          model_version='1', 
                                          image_path=None,
                                          mode=MODE)

    while True:
        if len(active_streams_dictionary) > 0:
            try:
                batched_images_yolov8n = []
                scr_names = []
                yolov8n_results = []
                for k, v in list(active_streams_dictionary.items()):
                    video_scr = k
                    video_stream_widget = v
                    model_type = video_stream_widget.get_model()
                    frame_num, manual_num, frame, frame_time = video_stream_widget.get_frame()  
                
                    if isinstance(video_scr, int): # src should be an integer is video source is camera
                        scr_name = str(video_scr)
                        frame_num = int(manual_num)
                    else:
                        scr_name = video_scr[-5:]

                    win_name = 'raw_frame_' + scr_name
                    file_name = win_name + '.jpg'
                    frame = cv2.resize(frame, (480, 300))
                    # cv2.imshow(win_name, frame)
                    cv2.imwrite(file_name, frame)
                    scr_names.append(scr_name)

                    if 'yolov8n' in model_type:
                        yolov8n_counter += 1
                        yolov8n_result_dict = {}
                        yolov8n_result_dict['model_type'] = 'yolov8n'     
                        yolov8n_result_dict['video_source'] = scr_name
                        yolov8n_result_dict['frame_num'] = frame_num
                        yolov8n_result_dict['frame_time'] = frame_time
                        yolov8n_result_dict['raw_frame'] = frame.copy()
                        yolov8n_results.append(yolov8n_result_dict)
                        batched_images_yolov8n.append(frame.copy()) 
                           
                
                # run yolov8s inference
                if len(batched_images_yolov8n) == 1:
                    infer_time = datetime.datetime.now()
                    _run_single_inference(yolov8n_wrapper, batched_images_yolov8n, scr_names, 'yolov8n', yolov8n_results)
                    result_time = datetime.datetime.now()
                    csvlogger.info([yolov8n_result_dict['model_type'],
                                    yolov8n_result_dict['video_source'],
                                    yolov8n_result_dict['frame_num'],
                                    yolov8n_result_dict['frame_time'],
                                    infer_time,
                                    result_time])
                  
                elif len(batched_images_yolov8n) > 1:
                    infer_time = datetime.datetime.now()
                    _run_batched_inference(yolov8n_wrapper, batched_images_yolov8n, scr_names, 'yolov8n', yolov8n_results)
                    result_time = datetime.datetime.now()
                    for i in range(yolov8n_results):
                        model_info = yolov8n_results[i]['model_type']
                        video_info = yolov8n_results[i]['video_source']
                        frame_num_info = yolov8n_results[i]['frame_num']
                        frame_time_info = yolov8n_results[i]['frame_time']
                        csvlogger.info([model_info, video_info, frame_num_info, frame_time_info, infer_time, result_time])
          
                
            except AttributeError:
                pass

        key = cv2.waitKey(1)
        if key == ord('q'):
            print('End running Yolov8n')
            cv2.destroyAllWindows()
            break
        else:
            pass

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global count
    count += 1
    rtsp_url = request.form['rtsp_url']
    model_type = request.form['model_type']
    model_type = model_type.split(",")

    # source of camera need to be an integer
    if len(rtsp_url) == 1:
        rtsp_url = int(rtsp_url)
    else:
        pass
    
    # invalidated model type should be rejected
    for m in model_type:
        if ('yolov8s' in m) or ('yolov8n' in m) or ('depth' in m):
            pass
        else:
            count -= 1
            return {"Bad request": "Wrong model type"}, 400

    url_list = list(active_streams_dictionary.keys())
    # if a stream already existed, check if the model already started or not
    if rtsp_url in url_list:
        count -= 1
        model_existed = model_list[url_list.index(rtsp_url)] # get the model that already existed
        model_existed_combined = '\t'.join(model_existed)
        for m in model_type:
            if m[:4] in model_existed_combined: # bad request if model already existed
                return {"Bad request": "Video stream already existed, and this type of model already started"}, 400
        else: 
            model_existed.append(m)
            model_type = model_existed
    else:
        pass

    if count <= int(NUM_STREAMS):
        video_stream_widget = VideoStream(src=rtsp_url, models=model_type)
        video_stream_widget.start()
        err_message = video_stream_widget.get_err_message()
        curr_model = video_stream_widget.get_model()
        if err_message is None:
            active_streams_dictionary[rtsp_url] = video_stream_widget
            if rtsp_url not in url_list:
                model_list.append(curr_model)
            print(list(active_streams_dictionary.keys()))
            print(model_list)
            print(count)
            return {"Stream started": rtsp_url}, 200
        else:
            count -= 1
            return {"Internal error": err_message}, 500
    else: 
        count -= 1
        return {"Bad request": "Exceed maximum video streams amount"}, 400
    

@app.route('/remove_stream', methods=['DELETE'])
def remove_stream():
    global count
    count -= 1
    rtsp_url = request.form['rtsp_url']

    if len(rtsp_url) == 1:
        rtsp_url = int(rtsp_url)
    else:
        pass

    url_list = list(active_streams_dictionary.keys())
    if rtsp_url in url_list:
        model_remove = model_list[url_list.index(rtsp_url)]
        model_list.remove(model_remove)
        for k, v in list(active_streams_dictionary.items()):
            if k == rtsp_url:
                curr_video_stream_widget = v
                curr_video_stream_widget.stop()
                del active_streams_dictionary[k] 
            else: 
                pass
        print(list(active_streams_dictionary.keys()))
        print(model_list)
        print(count)
    
        return {"Stream deleted": rtsp_url}, 200
    else:   
        count += 1          
        return {"Bad request": "Stream does not exist"}, 400
    

@app.route('/update_model', methods=['PATCH'])
def update_model():
    rtsp_url = request.form['rtsp_url']
    model_type_update = request.form['model_type']
    model_type_update = model_type_update.split(",")

    if len(rtsp_url) == 1:
        rtsp_url = int(rtsp_url)
    else:
        pass

    for m in model_type_update:
        if ('yolov8s' in m) or ('yolov8n' in m) or ('depth' in m):
            pass
        else:
            return {"Bad request": "Wrong model type"}, 400

    url_list = list(active_streams_dictionary.keys())
    if rtsp_url in url_list:
        model_index = url_list.index(rtsp_url)
        model_list[model_index] = model_type_update
        for k, v in list(active_streams_dictionary.items()):
            curr_video_stream_widget = v
            curr_video_stream_widget.update_model(model_type_update)
    else:
        return {"Bad request": "Stream does not exist"}, 400

    print(list(active_streams_dictionary.keys()))
    print(model_list)
    print(count)
    
    return {"Model updated": model_type_update}, 200

    
def run_flask_app():
    app.run(debug=False)


def get_yolo_outputs():
    while True:
        # if not yolo_results_queue.empty():
        yolo_results_dict = yolo_results_queue.get()
        video_source = yolo_results_dict['video_source']
        model_type = yolo_results_dict['model_type']
        frame_num = yolo_results_dict['frame_num']
        frame_time = yolo_results_dict['frame_time']
        print("Yolo test: ", 
            frame_num, 
            frame_time)


def get_depth_outputs():
    while True:
        # if not depth_results_queue.empty():
        depth_results_dict = depth_results_queue.get()
        video_source = depth_results_dict['video_source']
        model_type = depth_results_dict['model_type']
        frame_num = depth_results_dict['frame_num']
        frame_time = depth_results_dict['frame_time']

        print("Depth test: ", 
            frame_num, 
            frame_time)


if __name__ == '__main__':
    # initialize the models on a new thread
    # main_thread = Thread(target=initialize_model)
    yolov8s_thread = Thread(target=initialize_yolov8s)
    yolov8n_thread = Thread(target=initialize_yolov8n)
    depth_thread = Thread(target=initialize_depth)
    
    yolo_result_thread = Thread(target=get_yolo_outputs)
    depth_result_thread = Thread(target=get_depth_outputs)
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True

    # main_thread.start()
    yolov8s_thread.start()
    yolov8n_thread.start()
    depth_thread.start()
    flask_thread.start()
    yolo_result_thread.start()
    depth_result_thread.start()

    # main_thread.join()
    yolov8s_thread.join()
    yolov8n_thread.join()
    depth_thread.start()
    flask_thread.join()
    yolo_result_thread.join()
    depth_result_thread.start()

    # app.run(debug=True)
    # serve(app, host='0.0.0.0', port=5000)
