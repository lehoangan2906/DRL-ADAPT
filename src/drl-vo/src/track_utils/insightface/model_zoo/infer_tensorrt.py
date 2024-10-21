import time
from collections import namedtuple, OrderedDict

import tensorrt as trt
import cv2
import os
import numpy as np
import pycuda.driver as cuda
# import pycuda.autoinit
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

np.bool = np.bool_



class TensorRTModel:
    def __init__(self, engine_file_path, input_shape = None, output_shapes = None, init_net = True):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = self._load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()
        # Initialize CUDA
        cuda.init()
        device = cuda.Device(0)
        self.cuda_driver_context = device.make_context()
        # self.stream = cuda.Stream(0)
        self.input_shape,self.output_shapes = self.get_io_shapes()
        
        first_key = next(iter(self.input_shape))
        first_value = self.input_shape[first_key]
        self.input_shape = first_value

        # Allocate memory for inputs and outputs
        if init_net == False:
            self.input_shape = input_shape
            self.output_shapes = output_shapes
        
        # print(input_shape)
        # print(output_shapes)
        self.h_input = np.empty(self.input_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        
        self.h_outputs = [np.empty(shape, dtype=np.float32) for shape in self.output_shapes.values()]
        self.d_outputs = [cuda.mem_alloc(h_output.nbytes) for h_output in self.h_outputs]

    def _load_engine(self, engine_file_path):
        # with open(engine_file_path, "rb") as f:
        #     runtime = trt.Runtime(self.TRT_LOGGER)
        #     return runtime.deserialize_cuda_engine(f.read())
        with open(engine_file_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        return self.engine

    def infer(self, input_data):

        # Ensure the input data matches the expected input shape
        assert input_data.shape == self.input_shape, \
            f"Input data shape {input_data.shape} does not match expected shape {self.input_shape}"

        # Prepare input data
        np.copyto(self.h_input, input_data)

        # Push the CUDA context
        self.cuda_driver_context.push()

        # Transfer input data to the GPU
        # cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        cuda.memcpy_htod(self.d_input, self.h_input)

        # Execute the model
        bindings = [int(self.d_input)] + [int(d_output) for d_output in self.d_outputs]
        self.context.execute_v2(bindings=bindings)#, stream_handle=self.stream.handle

        # Transfer predictions back from GPU
        for h_output, d_output in zip(self.h_outputs, self.d_outputs):
            cuda.memcpy_dtoh(h_output, d_output)

        # Synchronize the stream
        # self.stream.synchronize()
        # Pop the CUDA context
        self.cuda_driver_context.pop()
        # Return the results
        return self.h_outputs

    def get_io_shapes(self):
        self.input_shapes = {}
        self.output_shapes = {}

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            if str(self.engine.get_tensor_mode(binding)) == "TensorIOMode.INPUT":
                self.input_shapes[binding] = shape
            else:
                self.output_shapes[binding] = shape

        return self.input_shapes, self.output_shapes

    def __getstate__(self):
        return {'model_path': self.engine_path}

    def __setstate__(self, state):
        self.__init__(state['model_path'])

    # def __del__(self):
    #     # Clean up
    #     if self.engine:
    #         self.engine.destroy()
    #     if self.context:
    #         self.context.destroy()

