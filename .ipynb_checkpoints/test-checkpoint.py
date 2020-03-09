from inference import Network
from process import post_processing
import cv2
import numpy as np
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def infer_on_stream():
    # Initialise the class
    infer_network = Network()
    infer_network.load_model("./models/mobilenet_ssd_pedestrian_detection/MobileNetSSD_deploy10695.xml", "CPU", CPU_EXTENSION)
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    img = cv2.imread('./frame1.jpg',0)
    
    img = cv2.resize(img, (net_input_shape[3],net_input_shape[2]))
    imgProcessed = img - 127.5
    imgProcessed = imgProcessed * 0.007843
    imgProcessed = imgProcessed.astype(np.float32)

    infer_network.exec_net(imgProcessed)
    if infer_network.wait() == 0:
        ### TODO: Get the results of the inference request ###
        width, height = imgProcessed.shape[:2]
        result = infer_network.get_output()
        h = img.shape[0]
        w = img.shape[1]
        box = result[0,0,:,3:7] * np.array([w, h, w, h])
        cls = result[0,0,:,1]
        conf = result[0,0,:,2]
        for i in range(len(box)):
            aR = abs(box[i][2] - box[i][0])*(box[i][3] - box[i][1])
            if conf[i] > 0.25 and aR<30000:
                cv2.rectangle(img, (int(box[i][0]), int(box[i][1])), (int(box[i][2]), int(box[i][3])), (0,255,0))
        cv2.imwrite("frameProcessed.jpg",img)        
#         print(box.astype(np.int32), conf, cls)
        
#         print(result)
#         for res in result:
#             if res[1] == 1 and res[2]>0.9:
#                 print(res[3],res[4],res[5],res[6])
   
        
def main():
    infer_on_stream()

if __name__ == '__main__':
    main()