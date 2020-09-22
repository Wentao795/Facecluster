#coding:utf-8
import os
import cv2
from mutildsbcan import dbscan
from mtcnn_detector import MtcnnDetector
import mxnet as mx
from model import MobileFaceNet
import torch
from Landmark2facefeture import Facefeature
import numpy as np
minPt = 200
distance = 0.50


def Video2list(videopath):
    imagelist = []
    videohanle = cv2.VideoCapture(videopath)
    success = True
    while success:
        success, frame_ = videohanle.read()
        if success:
            imagelist.append(frame_)
    return imagelist

def alignret(ret,len):
    bbox, point = ret
    bbox = bbox[0:len, :]
    point = point[0:len, :]
    return bbox,point
def main():
    #video load
    VideoPath = "../videos/3.mp4"
    imagelist = Video2list(VideoPath)

    #face detect
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(0), num_worker=1,minsize=80, accurate_landmark=True,
                             threshold=[0.6, 0.7, 0.9])
    Videoimginfo = []
    for img in imagelist:
        ret = detector.detect_face(img)
        Videoimginfo.append(ret)

    #face feature get
    model = MobileFaceNet(512)
    model_static_cnn = torch.load("model_mobilefacenet.pth", map_location=lambda storage, loc: storage)
    net_model_static_cnn = {}
    for k, v in model_static_cnn.items():
        if k=="fc2.weight":
            continue
        if k == "fc2.bias":
            continue
        net_model_static_cnn[k] = v
    model.load_state_dict(net_model_static_cnn)
    model.eval()
    imageinfo = []
    allFaceFeture = []
    for item in range(len(imagelist)):
        if Videoimginfo[item] is not None:
            image = imagelist[item]
            ret = Videoimginfo[item]
            facefeature = Facefeature(ret,image,model)
            imageinfo.append(len(facefeature[0]))
            allFaceFeture += facefeature[0]
            Videoimginfo[item] = [facefeature[1],facefeature[2]]
        else:
            imageinfo.append(0)

    Facecalsslist,classnum = dbscan(np.array(allFaceFeture),distance,minPt)
    print(Facecalsslist,classnum)

    #pic2video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    videoWrite = cv2.VideoWriter('output.avi', fourcc, 25, (imagelist[0].shape[1],imagelist[0].shape[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cc = 0
    flag =0
    for item in range(len(imageinfo)):
        img = imagelist[item]
        if imageinfo[item] == 0:
            videoWrite.write(img)
            cv2.imwrite("./ll/%d.jpg" % cc, img)
        else:
            #in this one pic may be has more than one pic
            # rectangle point lable ;
            bbox,point = Videoimginfo[item]
            for q in range(len(point)):
                for i in range(5):
                    cv2.circle(img, (int(point[q][i]), (int(point[q][i+5]))), 3, (0, 255, 0), -1)
                cv2.rectangle(img, (int(bbox[q][0]), int(bbox[q][1])), (int(bbox[q][2]), int(bbox[q][3])), (0, 255, 255), 2)
                cv2.putText(img,"%d"%Facecalsslist[flag],(int(bbox[q][0]), int(bbox[q][1])), font, 1.2, (255, 255, 255), 2)
                flag += 1
            cv2.imwrite("./ll/%d.jpg"%cc,img)
            videoWrite.write(img)
        cc+=1
    videoWrite.release()

if __name__ == "__main__":
    main()