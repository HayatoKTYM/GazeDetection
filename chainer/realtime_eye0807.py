import time
from chainer import Chain, Variable
start = time.time()

import argparse
import cv2
import os
from PIL import Image
import numpy as np
# coding: utf-8
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable
from chainer import cuda
from chainer import optimizers
from chainer.serializers import load_npz

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 50, 5)#50
            self.conv2 = L.Convolution2D(None, 50, 5)
            self.l1 = L.Linear(None, 100)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), 0.5)
        h = self.l2(h)
        return h

model = L.Classifier(CNN())
load_npz('eye0807.npz', model, strict=False)#Accuracy89%

np.set_printoptions(precision=2)
import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('/Users/hayato/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def predict(model, x_data):
    x = Variable(x_data.astype(np.float32))
    y = model.predictor(x)
    return np.argmax(y.data, axis = 1)

def getRep(bgrImg):
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        return None,None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    alignedFaces = []
    for box in bb:
        face = align.align(
            args.imgDim,
            rgbImg,
            box,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        alignedFaces.append(face[:32, :].astype(np.float32) / 255.)


    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    else:
        return (np.array(alignedFaces, dtype=np.float32), bb)


def infer(img, args):

    repsAndBBs = getRep(img)
    #repsAndBBs = image2TrainAndTest1(img)
    reps = repsAndBBs[0]
    bbs = repsAndBBs[1]
    persons = []
    confidences = []
    reps = reps.reshape(-1, 1, 32, 96)#

    if len(reps) == 0:
        return (confidences, bbs)
    y_pred = predict(model, reps)
    print(y_pred)

    return (y_pred,bbs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=400)
    parser.add_argument('--height', type=int, default=300)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--classifierModel',
        type=str,
        default='/home/shimada/openface/models/openface/celeb-classifier.nn4.small2.v1.pkl',
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)
    video_capture.set(5, args.fps)

    #out = cv2.VideoWriter('output/output.avi',
    #                          cv2.VideoWriter_fourcc('M','P', 'G' , '4'), args.fps,
    #                      (args.width, args.height))

    confidenceList = []
    while True:
        ret, frame = video_capture.read()
        confidences, bbs = infer(frame, args)
        #print (" C: " + str(confidences))
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            cv2.putText(frame, "{} ".format('NO FACE HERE'),
                        (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255 , 0, 255 ), 5)
            pass

        notations = []
        for i, c in enumerate(confidences):
            if c >= args.threshold:  # 0.5 is kept as threshold for known face.
                notations.append("non-looking")
            else:
                notations.append("looking")

        # Print the person name and conf value on the frame next to the person
        # Also print the bounding box
        for idx, person in enumerate(notations):
            #if person == "looking":
            cv2.rectangle(frame, (bbs[idx].left(), bbs[idx].top()),
                          (bbs[idx].right(), bbs[idx].bottom()),
                          (255 * int((confidences[idx])), 0,
                           255 * (1 - int(confidences[idx]))), 4)
            cv2.putText(frame, "{} ".format(person),
                        (bbs[idx].left(), bbs[idx].bottom() + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255*int(confidences[idx]), 0, 255*(1-int(confidences[idx]))), 2)

        cv2.imshow('eye_detection', frame)
        #out.write(frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    #out.release()
    cv2.destroyAllWindows()



