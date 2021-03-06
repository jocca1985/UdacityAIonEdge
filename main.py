"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from process import post_processing

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
PEOPLE_TOLERANCE=3
NO_PEOPLE_TOLERANCE=15

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocess(img,size):
    img = cv2.resize(img, size)
    imgProcessed = img - 127.5
    imgProcessed = imgProcessed * 0.007843
    imgProcessed = imgProcessed.astype(np.float32)
    imgProcessed = imgProcessed.transpose((2,0,1))
    imgProcessed = imgProcessed.reshape(1, 1, *imgProcessed.shape)
    return imgProcessed

def processResult(result, w, h):
    box = result[0,0,:,3:7] * np.array([w, h, w, h])
    cls = result[0,0,:,1]
    conf = result[0,0,:,2]
    return box,cls,conf


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

#     ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, CPU_EXTENSION)
    net_input_shape = infer_network.get_input_shape()

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Process frames until the video ends, or process is exited
    ### TODO: Loop until stream is over ###
    numFramesWithPerson = 0
    numFramesWithoutPerson = 0
    totalPersons = 0
    while cap.isOpened():
        # Read the next frame
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        ### TODO: Pre-process the image as needed ###
        p_frame = preprocess(frame, (net_input_shape[3], net_input_shape[2]))
        # Perform inference on the frame
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        # Get the output of inference
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            box, cls, conf = processResult(infer_network.get_output(), width, height)   
            resArray = []
            for i in range(len(box)):
                #check aspect ratio of the box
                aR = abs(box[i][2] - box[i][0])*(box[i][3] - box[i][1])
                if conf[i] > 0.25 and aR<90000:
                    resArray.append(box[i])
            if (len(resArray) > 0):
                numFramesWithPerson+=1
            else:
                numFramesWithoutPerson+=1

            if (numFramesWithPerson == PEOPLE_TOLERANCE and len(resArray)>0):
                numFramesWithoutPerson = 0
                ++totalPersons
                client.publish("person", json.dumps({"total": totalPersons}))
                 
                
            if (numFramesWithoutPerson == NO_PEOPLE_TOLERANCE and len(resArray)==0):
                client.publish("person/duration", json.dumps({"duration": int(numFramesWithPerson/24)})) 
                numFramesWithPerson = 0
            
            client.publish("person", json.dumps({"count": len(resArray)}))    
            
#             
            ## TODO: Extract any desired stats from the results ###
            
            ## TODO: Calculate and send relevant information on ###
            ## current_count, total_count and duration to the MQTT server ###
            ## Topic "person": keys of "count" and "total" ###
            ## Topic "person/duration": key of "duration" ###
            

        ## TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
