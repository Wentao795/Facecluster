import numpy as np
import cv2
from skimage import transform as trans
import torch
def mean_value(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ((image - 127.5) / 128.0)
    image = np.array(image, dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = image.reshape([-1,image.shape[0],image.shape[1],image.shape[2]])
    image = torch.tensor(image)
    return image

def Facefeature(ret,img,model):
    tpbbox = []
    tppoinst = []
    bbox, points = ret
    imgsize = [112,112]
    outputlist = []
    src = np.array([
        [30.2946+8, 51.6963],
        [65.5318+8, 51.5014],
        [48.0252+8, 71.7366],
        [33.5493+8, 92.3655],
        [62.7299+8, 92.2041]], dtype=np.float32)
    for q in range(len(points)):
        dst = np.array([
            [points[q][0], points[q][5]],
            [points[q][1], points[q][6]],
            [points[q][2], points[q][7]],
            [points[q][3], points[q][8]],
            [points[q][4], points[q][9]]], dtype=np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        if M is None:
            continue
        image = cv2.warpAffine(img, M, (imgsize[1], imgsize[0]), borderValue=0.0)
        model_output_fea1 = model(mean_value(image)).data.numpy().reshape(512)
        outputlist.append(model_output_fea1.tolist())
        tpbbox.append(bbox[q])
        tppoinst.append(points[q])
    return outputlist,tpbbox,tppoinst