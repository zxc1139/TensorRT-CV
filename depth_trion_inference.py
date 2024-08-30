import numpy as np
import os
import cv2
import sys
import time
import torch
import asyncio
import tritonclient.grpc as grpcclient
from functools import partial
from torchvision.transforms import Compose
from transform import Resize, NormalizeImage, PrepareForNet
import queue


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


class DepthTritonInference:
    def __init__(self, triton_client: str, model_name: str, model_version: str, image_path: str, mode: str):
        self.triton_client = triton_client
        self.model_name = model_name
        self.model_version = model_version
        self.image_path = image_path
        self.mode = mode
        if not self.triton_client.is_model_ready(model_name):
            print(model_name, "FAILED : is_model_ready")
            sys.exit(1)

        self.transform_depth = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=False, #True
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def depth_preprocessing(self, img, expected_width, expected_height):
        transform_depth = self.transform_depth
        input_image = cv2.resize(img, (expected_width, expected_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) / 255.0
        input_image = transform_depth({'image': input_image})['image']
        input_image = np.expand_dims(input_image, axis=0)
        return input_image


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

        (orig_h, orig_w) = original_image.shape[:2]
        input_image = self.depth_preprocessing(original_image, expected_width, expected_height)
        self.orig_h = orig_h
        self.orig_w = orig_w
        self.input_image = input_image
    

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
        ori_image_data = []
        if mode == 'image':
            for filename in filenames:
                img = cv2.imread(filename)
                input_image = self.depth_preprocessing(img, expected_width, expected_height)
                image_data.append(input_image)
                ori_image_data.append(img)
        
        if mode == 'video':
            for i in range(len(img_list)):
                if isinstance(img_list[i], dict):
                    img = img_list[i]['frame_data']
            
                else:
                    img = img_list[i]
                # img = img_list[i]
                input_image = self.depth_preprocessing(img, expected_width, expected_height)
                image_data.append(input_image)
                ori_image_data.append(img)

        self.filenames = filenames
        self.image_data = image_data
        self.ori_image_data = ori_image_data


    def send_async_requests(self, batch_size):
        # filenames = self.filenames
        image_data = self.image_data
        ori_image_data = self.ori_image_data
        triton_client = self.triton_client
        model_name = self.model_name
        model_version = self.model_version
        
        (orig_h, orig_w) = ori_image_data[0].shape[:2]

        requests = []
        responses = []
        request_ids = []
        image_idx = 0
        last_request = False
        async_requests = []
        sent_count = 0
        user_data = UserData()
        batched_ori_image = []
        final_outputs_lst  = []
        raw_depth_lst = []
        batch_size = batch_size
      
        start = time.time()
        while not last_request:
            repeated_image_data = []
            input_ori = []
            outputs = []

            for idx in range(batch_size):
                repeated_image_data.append(image_data[image_idx])
                input_ori.append(ori_image_data[image_idx])

                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True
            
            repeated_image_data = np.squeeze(repeated_image_data)
            batched_image_data = np.stack(repeated_image_data, axis=0)

            inputs = [grpcclient.InferInput('input', batched_image_data.shape, "FP32")] #images
            inputs[0].set_data_from_numpy(batched_image_data)
    
            outputs.append(grpcclient.InferRequestedOutput('output')) 
            batched_ori_image.append(input_ori)
            sent_count += 1

            async_requests.append(
                            triton_client.async_infer(
                            model_name=model_name,
                            inputs=inputs,
                            callback=partial(completion_callback, user_data),
                            request_id=str(sent_count),
                            model_version=model_version,
                            outputs=outputs,))


        processed_count = 0
        while processed_count < sent_count:
            (results, error) = user_data._completed_requests.get()
            processed_count += 1
            if error is not None:
                print("inference failed: " + str(error))
                sys.exit(1)
            responses.append(results)
        
        for results, ori_img in zip(responses, batched_ori_image):
            this_id = results.get_response().id
            depth_output = results.as_numpy('output')

            for i in range(batch_size):
                depth = torch.squeeze(torch.from_numpy(np.copy(depth_output[i, :])))
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.detach().cpu().numpy()
                depth = cv2.resize(depth, (orig_w, orig_h))
                depth = depth.astype(np.uint8)
                colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                final_outputs_lst.append(colored_depth)
                raw_depth_lst.append(depth)
            
        end = time.time()
        totalTime = end - start
        infer_fps = (batch_size * len(async_requests)) / totalTime
        return final_outputs_lst, raw_depth_lst, infer_fps



    def run_inference(self):
        start = time.time()
        triton_client = self.triton_client
        model_name = self.model_name
        model_version = self.model_version
        input_image = self.input_image
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('input', input_image.shape, "FP32"))
        # Initialize the data
        inputs[0].set_data_from_numpy(input_image)
        outputs.append(grpcclient.InferRequestedOutput('output'))
        results = triton_client.infer(model_name=model_name,
                                      model_version=model_version,
                                      inputs=inputs,
                                      outputs=outputs)
        depth_output = results.as_numpy('output')
        self.depth_output = depth_output
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        self.infer_fps = fps
        
    
    def save_output(self):
        depth = self.depth_output
        orig_w = self.orig_w
        orig_h = self.orig_h
        image_path = self.image_path
        mode = self.mode
        infer_fps = self.infer_fps
        # window_name = 'depth_' + str(src)

        depth = np.squeeze(np.array(np.copy(depth)))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (orig_w, orig_h))
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        if mode == 'image':
            f_name = image_path[7:-4]
            save_name = f_name + '_depth'
            cv2.imwrite('./data/' + save_name + '.jpg', colored_depth)
        if mode == 'video':
            cv2.putText(colored_depth, f'Inference Throughput: {int(infer_fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # return window_name, colored_depth 
            return colored_depth, depth
        
    def stop(self):
        sys.exit()
            



    