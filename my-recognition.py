#!/usr/bin/python3

import jetson_inference
import jetson_utils

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("output", type=str, help="filename of the image to save")
#parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.detectNet(network="ssd-mobilenet-v1", model="path to model", labels="/home/nih7/jetson-inference/python/training/detection/ssd/models/weapons-dataset-model/labels.txt", input_blob="input_0", output_cvg="scores", output_bbox="boxes")


orig_image = cv2.imread(opt.filename)

detections = net.Detect(img)
box0 = []
label0 = []
prob0 = []
for detection in detections:
    box0.append([detection.Left, detection.Top, detection.Right, detection.Bottom])
    label0.append(net.GetClassDesc(detection.ClassID))
    prob0.append(detection.Confidence)

for i in range(len(box0)):
    box = boxes[i]
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{label0[i]}: {prob0[i]:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = opt.output
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
