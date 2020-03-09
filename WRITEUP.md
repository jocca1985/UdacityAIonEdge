# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

If the network has custom layer that is not standard in OpenVino toolkit usually there are two options to deal with them. One is using Model Optimizer extensions and the other is to use particular optimizer (Caffe, TensorFlow, Onnx) to calculate output shape of each custom layer. 

Custom layers are needed for example when there is need to use some custom activation function that is not supported out of the box in OpenVino.

## Comparing Model Performance

For comparison of models I used accuracy for metrics (with fixed threshold). First tried extracting frames from the given example video and forming test set out of it. With the test created out of given video I tried measuring FP and TP rate. The model I chosen at the end had:
2% of false positives
96% of true positives with fixed threshold of 0.25.

Inference time was under 150ms which makes it convinient for execution on the Edge. Comparing to some GPU cloud service where the inference speed can go under 10ms cost benefit calculation is still on the Edge side since sending of each frame with best possible internet connection would take more than 500ms which makes cloud solution much slower. At the end hosting such solution would cost additional money for computing resources (on AWS for example).

The difference between model accuracy pre- and post-conversion was more less the same.

The size of the model pre- and post-conversion was the same.

The inference time of the model pre- and post-conversion was similar on CPU.


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
- detect number of people on some venue - for example some places needs to have limited amount of people and this app can help organizers to tell how many free places there are
- detect thiefs in shops - usually after working time there is only few security people who takes care of some shop...this app can detect possible anomalies
- statistics in metros - count how many people use metro in particular hour during the day


## Assess Effects on End User Needs

Different light conditions are usually the biggest problem in Computer Vision. In this case model usually perorm better in normal day light conditions. Size and quality of photo should also affect perofmace a lot.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Mobilenet_ssd_pedestrian_deteciton]
  - https://github.com/zlingkang/mobilenet_ssd_pedestrian_detection
  - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model MobileNetSSD_deploy10695.caffemodel --input_proto MobileNetSSD_deploy.prototxt

- This was the successfull model at the end.
  
- Model 2: [TinyYolo v2]
  - https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2
  - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model tiny_yolov2.onnx

  - The model was insufficient for the app because the it was not able to detect pedestrian on each frame
  - I tried to improve the model for the app by changing preprocessing and postprocessing.

