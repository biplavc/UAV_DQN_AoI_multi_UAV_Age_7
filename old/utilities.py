import os
from keras.losses import mean_squared_error
import numpy as np
import keras.backend as K
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF # biplav

class Tools:
    qfnet_data = 'DQN_data_xxx'
    qf_data = qfnet_data + '/qf_data' # each qlearning.py's start is run, the whole state is 14880 is stored here 
    qf_image = qfnet_data + '/qf_image'
    qf_model = qfnet_data + '/qf_model' # # each qlearning.py's start is run and finished, the whole q-table is stored here 
    ql_model_pro = qfnet_data + '/ql_model_pro'
    ql_model = qfnet_data + '/ql_model'

    @classmethod # https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod#:~:text=%40staticmethod%20function%20is%20nothing%20more,not%20Parent%20class%2C%20via%20inheritance.
    def create_dirs(cls):
        if not (os.path.exists(cls.qfnet_data)):
            os.mkdir(cls.qfnet_data)
        if not (os.path.exists(cls.qf_data)):
            os.mkdir(cls.qf_data)
        if not (os.path.exists(cls.qf_image)):
            os.mkdir(cls.qf_image)
        if not (os.path.exists(cls.qf_model)):
            os.mkdir(cls.qf_model)
        if not (os.path.exists(cls.ql_model_pro)):
            os.mkdir(cls.ql_model_pro)
        if not (os.path.exists(cls.ql_model)):
            os.mkdir(cls.ql_model)

def is_gpu_available():
    is_gpu = tf.test.is_gpu_available(True)
    return is_gpu

def is_gpu_available():
    is_gpu = tf.test.is_gpu_available(True)
    return is_gpu

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    gpu_name = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpu_name)  #  Return the number of GPUs

def set_gpu():
    if is_gpu_available():
        gpus = get_available_gpus()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(['{i}'.format(i=a) for a in range(gpus)]) # Visible to all GPUs

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # Set the GPU usage, but it will automatically allocate more when the demand is large

        # config.gpu_options.allow_growth=True # Do not occupy all of the video memory, allocate as needed

        session = tf.Session(config=config)
        KTF.set_session(session)

def str_map_float(str_array):
    nums = []
    for strs in str_array:
        nums.append(float(strs))
    return nums