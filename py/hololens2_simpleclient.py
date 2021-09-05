import socket
import struct
import abc
import threading
import time
from datetime import datetime, timedelta
from collections import namedtuple, deque
from enum import Enum
import numpy as np
import cv2

from utils import *

np.warnings.filterwarnings('ignore')

# Definitions
# Protocol Header Format
# see https://docs.python.org/2/library/struct.html#format-characters
VIDEO_STREAM_HEADER_FORMAT = "@qIIII18f"

VIDEO_FRAME_STREAM_HEADER = namedtuple(
    'SensorFrameStreamHeader',
    'Timestamp ImageWidth ImageHeight PixelStride RowStride fx fy '
    'PVtoWorldtransformM11 PVtoWorldtransformM12 PVtoWorldtransformM13 PVtoWorldtransformM14 '
    'PVtoWorldtransformM21 PVtoWorldtransformM22 PVtoWorldtransformM23 PVtoWorldtransformM24 '
    'PVtoWorldtransformM31 PVtoWorldtransformM32 PVtoWorldtransformM33 PVtoWorldtransformM34 '
    'PVtoWorldtransformM41 PVtoWorldtransformM42 PVtoWorldtransformM43 PVtoWorldtransformM44 '
)

RM_STREAM_HEADER_FORMAT = "@qIIII32f"

RM_FRAME_STREAM_HEADER = namedtuple(
    'SensorFrameStreamHeader',
    'Timestamp ImageWidth ImageHeight PixelStride RowStride '
    'rig2worldTransformM11 rig2worldTransformM12 rig2worldTransformM13 rig2worldTransformM14 '
    'rig2worldTransformM21 rig2worldTransformM22 rig2worldTransformM23 rig2worldTransformM24 '
    'rig2worldTransformM31 rig2worldTransformM32 rig2worldTransformM33 rig2worldTransformM34 '
    'rig2worldTransformM41 rig2worldTransformM42 rig2worldTransformM43 rig2worldTransformM44 '
    'sensorExtrinsicsM11 sensorExtrinsicsM12 sensorExtrinsicsM13 sensorExtrinsicsM14 '
    'sensorExtrinsicsM21 sensorExtrinsicsM22 sensorExtrinsicsM23 sensorExtrinsicsM24 '
    'sensorExtrinsicsM31 sensorExtrinsicsM32 sensorExtrinsicsM33 sensorExtrinsicsM34 '
    'sensorExtrinsicsM41 sensorExtrinsicsM42 sensorExtrinsicsM43 sensorExtrinsicsM44 '
)

# Each port corresponds to a single stream type
VIDEO_STREAM_PORT = 23940
AHAT_STREAM_PORT = 23941
LEFT_FRONT_STREAM_PORT = 23942
RIGHT_FRONT_STREAM_PORT = 23943

HOST = '169.254.189.82' #'169.254.189.82' #'192.168.1.242'
#'192.168.1.92'

HundredsOfNsToMilliseconds = 1e-4
MillisecondsToSeconds = 1e-3


class SensorType(Enum):
    VIDEO = 1
    AHAT = 2
    LONG_THROW_DEPTH = 3
    LF_VLC = 4
    RF_VLC = 5


class FrameReceiverThread(threading.Thread):
    def __init__(self, host, port, header_format, header_data):
        super(FrameReceiverThread, self).__init__()
        self.header_size = struct.calcsize(header_format)
        self.header_format = header_format
        self.header_data = header_data
        self.host = host
        self.port = port
        self.latest_frame = None
        self.latest_header = None
        self.socket = None

    def get_data_from_socket(self, lut = False):
        # read the header in chunks
        reply = self.recvall(self.header_size)

        if not reply:
            print('ERROR: Failed to receive data from stream.')
            return

        data = struct.unpack(self.header_format, reply)
        header = self.header_data(*data)

        # read the image in chunks
        all_bytes = 0
        image_size_bytes = header.ImageHeight * header.RowStride

        size_of_float = 4
        lut_bytes = header.ImageHeight * header.ImageWidth * 3 * size_of_float

        all_bytes += image_size_bytes
        all_bytes += lut_bytes if lut else 0

        all_data = self.recvall(all_bytes)

        if lut:
            image_data = all_data[lut_bytes:]
            lut_data = all_data[:lut_bytes]
            return header, image_data, lut_data

        return header, all_data

    def recvall(self, size):
        msg = bytes()
        while len(msg) < size:
            newsize = size - len(msg)
            try:
                part = self.socket.recv(newsize) if newsize > 0 else ''
            except:
                part = ''
            if part == '':
                break  # the connection is closed
            msg += part
        return msg

    def start_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        
        # send_message(self.socket, b'socket connected at ')
        print('INFO: Socket connected to ' + self.host + ' on port ' + str(self.port))

    def start_listen(self):
        t = threading.Thread(target=self.listen)
        t.start()

    @abc.abstractmethod
    def listen(self):
        return

    @abc.abstractmethod
    def get_mat_from_header(self, header):
        return


class VideoReceiverThread(FrameReceiverThread):
    def __init__(self, host):
        super().__init__(host, VIDEO_STREAM_PORT, VIDEO_STREAM_HEADER_FORMAT,
                         VIDEO_FRAME_STREAM_HEADER)

    def listen(self):
        while True:
            self.latest_header, image_data = self.get_data_from_socket()
            self.latest_frame = np.frombuffer(image_data, dtype=np.uint8).reshape((self.latest_header.ImageHeight,
                                                                                   self.latest_header.ImageWidth,
                                                                                   self.latest_header.PixelStride))

    def get_mat_from_header(self, header):
        pv_to_world_transform = np.array(header[7:24]).reshape((4, 4)).T
        return pv_to_world_transform


class AhatReceiverThread(FrameReceiverThread):
    def __init__(self, host, port):
        super().__init__(host,
                         port, RM_STREAM_HEADER_FORMAT, RM_FRAME_STREAM_HEADER)

    def listen(self):
        while True:
            self.latest_header, image_data, lut_data = self.get_data_from_socket(True)
            self.latest_frame = np.frombuffer(image_data, dtype=np.uint16).reshape((self.latest_header.ImageHeight,
                                                                                    self.latest_header.ImageWidth))
            self.lut = np.frombuffer(lut_data, dtype=np.float32).reshape((self.latest_header.ImageHeight * self.latest_header.ImageWidth,
                                                                          3))

    def get_mat_from_header(self, header):
        rig_to_world_transform = np.array(header[5:22]).reshape((4, 4)).T
        return rig_to_world_transform 

class SpatialCamsReceiverThread(FrameReceiverThread):
    def __init__(self, host, port):
        super().__init__(host,
                         port, RM_STREAM_HEADER_FORMAT, RM_FRAME_STREAM_HEADER)

    def listen(self):
        while True:
            self.latest_header, image_data = self.get_data_from_socket(True)
            self.latest_frame = np.frombuffer(image_data, dtype=np.uint8).reshape((self.latest_header.ImageHeight,
                                                                                    self.latest_header.ImageWidth))

    def get_mat_from_header(self, header):
        rig_to_world_transform = np.array(header[5:22]).reshape((4, 4)).T
        return rig_to_world_transform 

if __name__ == '__main__':
    video_receiver = VideoReceiverThread(HOST)
    video_receiver.start_socket()

    ahat_receiver = AhatReceiverThread(HOST, AHAT_STREAM_PORT)
    ahat_receiver.start_socket()

    lf_receiver = SpatialCamsReceiverThread(HOST, LEFT_FRONT_STREAM_PORT)
    # lf_receiver.start_socket()
    
    rf_receiver = SpatialCamsReceiverThread(HOST, RIGHT_FRONT_STREAM_PORT)
    # rf_receiver.start_socket()

    video_receiver.start_listen()
    ahat_receiver.start_listen()
    # lf_receiver.start_listen()
    # rf_receiver.start_listen()

    start_recording = False
    save_one = 0
    while True:
        if np.any(video_receiver.latest_frame):
            cv2.imshow('Photo Video Camera Stream', video_receiver.latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if np.any(ahat_receiver.latest_frame):
            cv2.imshow('Depth Camera Stream', ahat_receiver.latest_frame)

            # Get xyz points in camera space
            points = get_points_in_cam_space(ahat_receiver.latest_frame, ahat_receiver.lut)
            
            output_path = "C:/Users/halea/Documents/test" + str(save_one) + ".ply"
            save_one += 1
            
            save_ply(output_path, points, rgb=None)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(33) == ord('r'):
            start_recording = not start_recording

            if start_recording:
                print("start recording")
                front_left_vid = cv2.VideoWriter('front-left.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, (640,480))
                front_right_vid = cv2.VideoWriter('front-right.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, (640,480))
            else: 
                print("stop recording")
                front_left_vid.release()
                front_right_vid.release()    

        if np.any(lf_receiver.latest_frame):
            cv2.imshow('Left Front Camera Stream', lf_receiver.latest_frame)
            
            if start_recording:
                color = cv2.cvtColor(lf_receiver.latest_frame, cv2.COLOR_GRAY2BGR)
                front_left_vid.write(color)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if np.any(rf_receiver.latest_frame):
            cv2.imshow('Right Front Camera Stream', rf_receiver.latest_frame)

            if start_recording:
                color = cv2.cvtColor(rf_receiver.latest_frame, cv2.COLOR_GRAY2BGR)
                front_right_vid.write(color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break