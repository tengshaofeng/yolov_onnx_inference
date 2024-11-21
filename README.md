# yolov_onnx_inference
yolo各个版本（只测试了yolov10x和yolo11x）的onnx推理代码

# 步骤一：
## 下载yolo各个版本的pt模型
### yolo11 : 
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{x|l|m|s|n}.pt # float16版本
### yolov10: 
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10{n/s/m/b/l/x}.pt  # float16版本

wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt  # float32版本

# 步骤二：
## Export the model to ONNX format
path = model.export(format="onnx", half=True)

gpu机器下跑model.export(format="onnx", half=True, device=0),onnx模型size才会小于等于pt模型
# 步骤三：
python inference_onnx.py

参考：https://github.com/ultralytics/ultralytics
