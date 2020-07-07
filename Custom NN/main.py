import cv2
import sys
import os
import time
import argparse
import requests
import numpy as np
from numpy import loadtxt
import csv
from detectFace import FaceDetector
from utility import annotate_imageX
import matplotlib.pyplot as plt


Path_images = []

Sofa3 = "Sofa3.txt"
Sofa2 = "Sofa2.txt"
Sofa1 = "Sofa1.txt"
NoFace = "NoFace.txt"

Images_SelfDataSet = []
DetectedFaces = []
BOXES = []

ImageSets = [Sofa3]

pathTOSAVE = "RRES"
Loc = "result_"


files =[]
boxArrau = []
vals = []
obj_thresh = []
nb_boxarray = []
fileToOpen = "FDDB_complete"
ResultFile = "FDDB_CustomNN_result"





def getWEbCam(thresh):
    face_detector = FaceDetector()
    predictedCoords = np.zeros((2, 1), np.float32)
    cap = cv2.VideoCapture(0)
    centers = []
    # cv2.namedWindow("window", cv2.WND)

    now = time.time()
    while cap.isOpened():
        now = time.time()
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            break

        rgv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if thresh:
            bboxes = face_detector.predict(rgv_frame, thresh)
        else:
            bboxes = face_detector.predict(rgv_frame)
        ann_frame = annotate_image(frame, bboxes)
        cv2.imshow('window', ann_frame)
        # print(bboxes)

        print("FPS: {:0.2f}".format(1 / (time.time() - now)), end="\r", flush=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def getSource(thresh):
    face_detector = FaceDetector()
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    now = time.time()
    stream = requests.get('http://gkiaxis.informatik.privat/axis-cgi/mjpg/video.cgi?resolution=4CIF&color=1',auth=('gki', 'Felix&2'), stream=True)
    if (stream.status_code == 200):
        print("================================================================================")
        print("++++++++++++++Connected to network Camera, Now creating a pipeline++++++++++++++")
        print("================================================================================")
        bytesX = bytes()
        for chunk in stream.iter_content(chunk_size = 1024):
            bytesX += chunk
            a = bytesX.find(b'\xff\xd8')
            b = bytesX.find(b'\xff\xd9')
            if a!= -1 and b != -1:
                jpg = bytesX[a:b + 2]
                bytesX = bytesX[b + 2:]
                frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                image_copy = frame
                if frame.shape[0] == 0:
                    break
                rgv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if thresh:
                    bboxes = face_detector.predict(rgv_frame, thresh)
                else:
                    bboxes = face_detector.predict(rgv_frame)
                # print(bboxes)
                # del bboxes[:1]
                ann_frame = annotate_image(frame, bboxes)
                # frame = Grid(ann_frame)
                cv2.imshow('window', ann_frame)
                # print("FPS: {:0.2f}".format(1 / (time.time() - now)), end="\r", flush=True)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

def Grid(frameIn):
    line_color = (255,0,0)
    thickness = 1
    type_ = cv2.LINE_AA
    pxstep = 150
    frame = frameIn
    x = pxstep
    y = pxstep
    while x < frame.shape[1]:
        cv2.line(frame, (x,0),  (x, frame.shape[0]), color=line_color, lineType = type_, thickness = thickness)
        x += pxstep
    while y <frame.shape[0]:
        cv2.line(frame, (0,y),  (frame.shape[1], y), color=line_color, lineType= type_, thickness = thickness)
        y += pxstep

    return frame


def CustomData():
    face_detector = FaceDetector()
    Bounds = np.load("Bounds.npy")
    totalLen = 101
    with open("DataSet_folds.csv","r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            path = str(row[0])
            pathCopy = path
            XXXNames = pathCopy.split("\\",1)[1]
            # print(XXXNames)
            numBoxes = int(row[1])
            rects = Bounds[line_count]
            image  = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            bboxes = face_detector.predict(image, 0.8)
            ann_frame,centers = annotate_imageX(image, bboxes)
            for boxes in rects:
                xmin = boxes[0][0]
                ymin = boxes[0][1]
                xmax = boxes[1][0]
                ymax = boxes[1][1]
                cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0,0,255), thickness=3)
                cv2.imwrite(os.path.join(pathTOSAVE , XXXNames), ann_frame)
            line_count += 1
            cv2.imshow(path, image)
            cv2.waitKey()
            cv2.destroyAllWindows()

    # print my_string.split("world",1)[1] 

    # for index, fileName in enumerate(ImageSets):
    #     with open(fileName, 'r') as f:
    #         for fileS in f:
    #             Images_SelfDataSet.append(fileS)
    #         print("All appended")
    #     for indasdex, path in enumerate(Images_SelfDataSet):
    #         path = path.strip()
    #         image  = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #         # rgv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         bboxes = face_detector.predict(image, 0.8)
    #         ann_frame,centers = annotate_imageX(image, bboxes)
    #         cv2.imwrite(os.path.join(pathTOSAVE , Loc+str(indasdex)+'.jpg'), ann_frame)
    #         # DetectedFaces.append(bboxes)
    #         BOXES.append(centers)
    #         DetectedFaces.append(len(bboxes))
    #     DataSETTEXT(fileName, Images_SelfDataSet, BOXES, DetectedFaces)

def DataSETTEXT(fileName, ImageSet, BOXES, DetectedFaces):
    with open(fileName+"_results.txt", "w") as f:
        for index, path in enumerate(ImageSet):
            print("Currently Looping on ", index)
            f.write(path.strip()+"\t"+str(DetectedFaces[index])+"\t"+str(BOXES[index])+"\n")
            # print(path.strip()+"\t"+str(DetectedFaces[index])+"\t"+str(BOXES[index]))
        f.close()
    return 0

def doFDDBEval(thresh):
    with open(fileToOpen+".txt", 'r') as f:
        for file in f:
            Path_images.append(file)
        print("All appended")
    
    face_detector = FaceDetector()
    for path_orig in Path_images:
        path = path_orig.strip() + ".jpg"
        image  = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        rgv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = face_detector.predict(rgv_frame, thresh)
        nb_boxarray.append(len(bboxes))
        files.append(path_orig.strip())
        obj_thresh.append(0.8)
        if len(bboxes) > 0:
            boxArrau.append(bboxes)
        else:
            boxArrau.append([])
            
    save_to_text(files,nb_boxarray,boxArrau)
        
        # ann_frame = annotate_image(rgv_frame, bboxes)
        
        # cv2.imshow('Test image',ann_frame)
        # cv2.waitKey()


# def doFDDBEval(thresh):
    # with open(fileToOpen+".txt", 'r') as f:
    #     for file in f:
    #         Path_images.append(file)
    #     print("All appended")

#     face_detector = FaceDetector()
#     predictedCoords = np.zeros((2, 1), np.float32)
#     centers = []

#     for pathX in Path_images:
#         path = pathX.strip() + ".jpg"
#         img = cv2.imread(path)
#         # cv2.resize(img,(640,480))
#         rgv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         bboxes = face_detector.predict(rgv_frame, thresh)
#         ann_frame = annotate_image(rgv_frame, bboxes)
#         nb_boxarray.append(len(bboxes))
#         files.append(pathX.strip())
#         if len(bboxes) != 0:
#             boxArrau.append(bboxes)                
#             obj_thresh.append(0.8)

#         else:
#             boxArrau.append([])
#             obj_thresh.append(0.8)
#         s =input("ssdsds")    
#         cv2.imshow('window', ann_frame)
#         x =input("s")

#     save_to_text(files,nb_boxarray,boxArrau)
#     write2txt()


# files =[]
# boxArrau = []
# vals = []
# obj_thresh = []
# nb_boxarray = []

def GetFromFiles(thresh):
    with open("PATH.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            Actualpath = line.strip()+".jpg"
            files.append(Actualpath)

    with open("nb_boxarray.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            ActualVal = line.strip()
            nb_boxarray.append(ActualVal)

    boxArrau = np.load("boxArraYNUMPYU.npy")


    # PATH = np.load("PATH.npy")
    # for element in PATH:
    #     print(element, type(element))

    # nb_boxarray = np.load("nb_boxarray.npy")
    # for element in nb_boxarray:
    #     print(element, type(element))

    with open(ResultFile+".txt", "w") as f:
        for index, path_orig in enumerate(files):
            print("Currently on :", index, path_orig)
            f.write(path_orig.strip()+'\n')
            f.write(str(nb_boxarray[index])+'\n')
            path = path_orig.strip()
            for box in boxArrau[index]:
                x1 = int(box[0] - box[2]/2)
                y1 = int(box[1] - box[3]/2)
                x2 = int(box[0] + box[2]/2)
                y2 = int(box[1] + box[3]/2)
                w = x2 - x1
                h = y2 - y1

                print(x1,y1,w,h)
                # x = input("asdasd")
                f.write(str(x1)+' '+str(y1)+' '+str(w)+' '+str(h)+' '+str(1)+'\n')


            # image  = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # ann_frame = annotate_image(image, boxArrau[index])
            # cv2.imshow('Test image', ann_frame)
            # cv2.waitKey()






def write2txt():
    labels = ['face']
    quiet = True
    with open(ResultFile+".txt", "w") as f:
        for index, path in enumerate(files):
            print("Currently on ", index, path)
            f.write(path.strip()+'\n')
            f.write(str(nb_boxarray[index])+'\n')
            toloppover = boxArrau[index]
            x = obj_thresh[index]
            path = path.strip() + ".jpg"
            img = cv2.imread(path)
            x = input("asd")
            ann_frame = annotate_image(img, toloppover)
            cv2.imshow('window', ann_frame)
            # if nb_boxarray[index] > 0:
            #     for box in toloppover:
            #         print(box, type(box), len(box))
            #         x = input("asd")
            #         f.write(str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(1)+'\n')
                    # for x,y,w,h,p in box:
                    # cv2.rectangle(ret, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 3)
        return 0


def save_to_text(path,nb_boxarray,boxArrau):
    # with open("PATH.txt","w") as f:
    #     for index, path in enumerate(path):
    #         f.write(path.strip()+'\n')

    # with open("nb_boxarray.txt","w") as f:
    #     for element in nb_boxarray:
    #         f.write(str(element)+"\n")
                
    # with open("boxArrau.txt","w") as f:
    #     for arXr in boxArrau:
    #         f.write(str(arXr)+'\n')

    np.save("boxArraYNUMPYU.npy",boxArrau)
    np.save("PATH.npy",boxArrau)
    np.save("nb_boxarray.npy",boxArrau)



if __name__ == "__main__":
    argument = sys.argv[1]


    if(argument == "fddb"):
        doFDDBEval(0.8)

    
    if(argument == "gf"):
        GetFromFiles(0.8)
        
    if(argument == "webcam"):
        getWEbCam(0.9)

    if(argument == "haar"):
        haarAlgo()

    if (argument == "custom"):
        CustomData()