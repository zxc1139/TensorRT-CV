import numpy as np
import cv2
import sys
import os
import torch
import time
import tritonclient.grpc as grpcclient
from config import CLASSES, COLORS
from torch_utils import det_postprocess
from utils import blob, letterbox, parse_model, path_to_list
from tritonclient.utils import InferenceServerException
from functools import partial
import queue

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def yolo_preprocessing(img, expected_width, expected_height):
    bgr, ratio, dwdh = letterbox(img, (expected_width, expected_height))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    input_image = blob(rgb, return_seg=False)
    dwdh = np.asarray(dwdh * 2, dtype=np.float32)
    input_image = input_image.astype(np.float32)
    return input_image, ratio, dwdh


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))



class YoloTritonInference(object):
    def __init__(self, triton_client: str, model_name: str, model_version: str, image_path: str, mode: str):
        self.triton_client = triton_client
        self.model_name = model_name
        self.model_version = model_version
        self.image_path = image_path
        self.mode = mode
        if not self.triton_client.is_model_ready(model_name):
            print(model_name, "FAILED : is_model_ready")
            sys.exit(1)

        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=model_name, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        try:
            model_config = triton_client.get_model_config(
                model_name=model_name, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)



    def read_image(self, raw_frame=None):
        image_path = self.image_path
        triton_client = self.triton_client
        model_name = self.model_name
        mode = self.mode
        expected_image_shape = triton_client.get_model_metadata(model_name).inputs[0].shape[-2:]
        expected_width = expected_image_shape[0]
        expected_height = expected_image_shape[1]

        if mode == 'image':
            original_image: np.ndarray = cv2.imread(image_path)
        if mode == 'video':
            original_image = raw_frame

        input_image, ratio, dwdh = yolo_preprocessing(original_image, expected_width, expected_height)
        self.original_image = original_image
        self.input_image = input_image
        self.dwdh = dwdh
        self.ratio = ratio


    
    def read_image_stack(self, folder_dir, mode, img_list=None):
        triton_client = self.triton_client
        model_name = self.model_name
        mode = self.mode
        expected_image_shape = triton_client.get_model_metadata(model_name).inputs[0].shape[-2:]
        expected_width = expected_image_shape[0]
        expected_height = expected_image_shape[1]

        if mode == 'image':
            if os.path.isdir(folder_dir):
                filenames = [
                os.path.join(folder_dir, f)
                for f in os.listdir(folder_dir)
                if os.path.isfile(os.path.join(folder_dir, f))]
            else:
                filenames = [folder_dir]
            filenames.sort()
        else:
            filenames = None
        
        image_data = []
        ratio_data = []
        dwdh_data = []
        ori_image_data = []
        if mode == 'image':
            for filename in filenames:
                img = cv2.imread(filename)
                input_image, ratio, dwdh = yolo_preprocessing(img, expected_width, expected_height)
                image_data.append(input_image)
                ratio_data.append(ratio)
                dwdh_data.append(dwdh)
                ori_image_data.append(img)
        
        if mode == 'video':
            for i in range(len(img_list)):
                if isinstance(img_list[i], dict):
                    img = img_list[i]['frame_data']
                else:
                    img = img_list[i]
                input_image, ratio, dwdh = yolo_preprocessing(img, expected_width, expected_height)
                image_data.append(input_image)
                ratio_data.append(ratio)
                dwdh_data.append(dwdh)
                ori_image_data.append(img)

        self.filenames = filenames
        self.image_data = image_data
        self.ratio_data = ratio_data
        self.dwdh_data = dwdh_data
        self.ori_image_data = ori_image_data


    def send_async_requests(self, batch_size):
        # filenames = self.filenames
        image_data = self.image_data
        ratio_data = self.ratio_data
        dwdh_data = self.dwdh_data
        ori_image_data = self.ori_image_data
        triton_client = self.triton_client
        model_name = self.model_name
        model_version = self.model_version

        # requests = []
        responses = []
        # result_filenames = []
        # request_ids = []
        image_idx = 0
        last_request = False
        async_requests = []
        sent_count = 0
        batch_size = batch_size
        user_data = UserData()
        batched_ratio = []
        batched_dwdh = []
        batched_ori_image = []
        final_outputs_lst = []
        batched_bboxes_lst = []
        batched_scores_lst = []
        batched_labels_lst = []
        batched_classes_lst = []


        start = time.time()
        while not last_request:
            input_filenames = []
            repeated_image_data = []
            input_ratio = []
            input_dwdh = []
            input_ori = []
            outputs = []

            for idx in range(batch_size):
                # input_filenames.append(filenames[image_idx])
                repeated_image_data.append(image_data[image_idx])
                input_ratio.append(ratio_data[image_idx])
                input_dwdh.append(dwdh_data[image_idx])
                input_ori.append(ori_image_data[image_idx])

                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True
            
            repeated_image_data = np.squeeze(repeated_image_data)
            batched_image_data = np.stack(repeated_image_data, axis=0)
            
            # Send request
            inputs = [grpcclient.InferInput('images', batched_image_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(batched_image_data)
            
            outputs.append(grpcclient.InferRequestedOutput('num_dets'))
            outputs.append(grpcclient.InferRequestedOutput('bboxes'))
            outputs.append(grpcclient.InferRequestedOutput('scores'))
            outputs.append(grpcclient.InferRequestedOutput('labels'))    
            batched_ratio.append(input_ratio)
            batched_dwdh.append(input_dwdh)
            batched_ori_image.append(input_ori)
            sent_count += 1
 
            async_requests.append(
                            triton_client.async_infer(
                            model_name=model_name,
                            model_version=model_version,
                            inputs=inputs,
                            callback=partial(completion_callback, user_data),
                            request_id=str(sent_count),
                            outputs=outputs,))

        
        processed_count = 0
        while processed_count < sent_count:
            (results, error) = user_data._completed_requests.get()
            processed_count += 1
            if error is not None:
                print("inference failed: " + str(error))
                sys.exit(1)
            responses.append(results)
        
        for results, ratio, dwdh, ori_img in zip(responses, batched_ratio, batched_dwdh, batched_ori_image):
            this_id = results.get_response().id
            num_dets = results.as_numpy('num_dets')
            bboxes = results.as_numpy('bboxes')
            scores = results.as_numpy('scores')
            labels = results.as_numpy('labels')
        
            for i in range(batch_size):
                curr_num_dets = torch.squeeze(torch.from_numpy(np.copy(num_dets[i, :])))
                curr_bboxes = torch.squeeze(torch.from_numpy(np.copy(bboxes[i, :, :])))
                curr_scores = torch.squeeze(torch.from_numpy(np.copy(scores[i, :])))
                curr_labels = torch.squeeze(torch.from_numpy(np.copy(labels[i, :])))
                curr_ratio = ratio[i]
                curr_dwdh = dwdh[i]
                curr_img = ori_img[i]

                keep_bboxes, keep_scores, keep_labels = det_postprocess(
                    curr_num_dets, curr_bboxes, curr_scores, curr_labels)
                keep_bboxes -= curr_dwdh
                keep_bboxes /= curr_ratio
                keep_classes = []

                for (bbox, score, label) in zip(keep_bboxes, keep_scores, keep_labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    cls = CLASSES[cls_id]
                    keep_classes.append(cls)
                    color = COLORS[cls]
                    if score >= 0.25:
                        cv2.rectangle(curr_img, bbox[:2], bbox[2:], color, 2)
                        cv2.putText(curr_img,
                                    f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, [225, 255, 255],
                                    thickness=2)
                final_outputs_lst.append(curr_img)
                batched_bboxes_lst.append(keep_bboxes)
                batched_scores_lst.append(keep_scores)
                batched_labels_lst.append(keep_labels)
                batched_classes_lst.append(keep_classes)

        end = time.time()
        totalTime = end - start
        infer_fps = (batch_size * len(async_requests)) / totalTime
        return final_outputs_lst, batched_bboxes_lst, batched_scores_lst, batched_labels_lst, batched_classes_lst, infer_fps


    def run_inference(self):
        start = time.time()
        model_name = self.model_name
        model_version = self.model_version
        triton_client = self.triton_client
        input_image = self.input_image
        user_data = UserData()
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('images', input_image.shape, "FP32"))
        # Initialize the data
        inputs[0].set_data_from_numpy(input_image)
        outputs.append(grpcclient.InferRequestedOutput('num_dets'))
        outputs.append(grpcclient.InferRequestedOutput('bboxes'))
        outputs.append(grpcclient.InferRequestedOutput('scores'))
        outputs.append(grpcclient.InferRequestedOutput('labels'))
        
        # Test with outputs
        results = triton_client.infer(model_name=model_name,
                                      model_version=model_version,
                                      inputs=inputs,
                                      outputs=outputs)


        num_dets = results.as_numpy('num_dets')
        bboxes = results.as_numpy('bboxes')
        scores = results.as_numpy('scores')
        labels = results.as_numpy('labels')
        self.num_dets = num_dets
        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        self.infer_fps = fps

        
        
    def save_output(self):
        num_dets = self.num_dets
        bboxes = self.bboxes
        scores = self.scores
        labels = self.labels
        dwdh = self.dwdh
        ratio = self.ratio
        original_image = self.original_image
        image_path = self.image_path
        mode = self.mode
        infer_fps = self.infer_fps
        # window_name = 'yolo_' + str(src)

        num_dets = torch.squeeze(torch.from_numpy(np.copy(num_dets)))
        scores = torch.squeeze(torch.from_numpy(np.copy(scores)))
        labels = torch.squeeze(torch.from_numpy(np.copy(labels)))
        bboxes = torch.squeeze(torch.from_numpy(np.copy(bboxes)))

        keep_bboxes, keep_scores, keep_labels = det_postprocess(num_dets, bboxes, scores, labels)
        keep_bboxes -= dwdh
        keep_bboxes /= ratio
        keep_classes = []

        for (bbox, score, label) in zip(keep_bboxes, keep_scores, keep_labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            keep_classes.append(cls)
        
            if score >= 0.25:
                cv2.rectangle(original_image, bbox[:2], bbox[2:], color, 2)
                cv2.putText(original_image,
                            f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)
                    
        if mode == 'image':
            f_name = image_path[7:-4]
            save_name = f_name + '_yolo'
            cv2.imwrite('./data/' + save_name + '.jpg', original_image)
        if mode == 'video':
            cv2.putText(original_image, f'Inference Throughput: {int(infer_fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            return original_image, keep_bboxes, keep_scores, keep_labels, keep_classes    

    
                
    
        
