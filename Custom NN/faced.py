import cv2
from utility import annotate_image
import os
from detectFace import FaceDetector
from utility import annotate_image



face_detector = FaceDetector()
ResultFile = "customYOLO.txt"

image_paths = []
boxArray = []
number_of_boxes = []

parent_directory = "originalPics/"

with open("filePath.txt", "r") as f:
        for line in f:
                paths = parent_directory+line.strip()+".jpg"
                img = cv2.imread(paths)
                bboxes = face_detector.predict(img)
                image_paths.append(line.strip())
                if(len(bboxes) > 0):
                        boxArray.append(bboxes)
                        number_of_boxes.append(len(bboxes))
                if(len(bboxes) == 0):
                        tupple = (0,0,0,0) 
                        boxArray.append([tupple])
                        number_of_boxes.append(0)
        labels = ['face']
        quiet = True
        with open(ResultFile, "w") as f:
                for index, path in enumerate(image_paths):
                # print("Currently Looping on ", index)
                        f.write(path.strip()+'\n')
                        f.write(str(number_of_boxes[index])+'\n')
                        if number_of_boxes[index] > 0:
                                toloppover = boxArray[index]
                                p = boxArray[index]
                                x = len(toloppover)
                                if x > 0:
                                        tupLen = x
                                        for box in p:
                                                f.write(str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(1)+'\n')         
      
