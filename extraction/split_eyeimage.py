#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__ = "Hayato Katayama"
__date__    = "20190304"

import cv2, os, glob, csv
import argparse
import openface
import numpy as np

def split_eyeimage(folder):
    """

    画像から顔を検出して切り出す(保存)
    目の部分を切り取る(保存)

    @param: folder あるユーザの画像PATHが格納されたフォルダ

    同directory内のfaceフォルダに顔画像を保存する(96*96)
    同directory内のeyeフォルダに顔画像を保存する(96*32)
    同directory内のlandmarkフォルダにPATH:landmarkのペアを保存する
    """
    PATHS = []

    print(folder)
    path = folder.replace('out', 'face', 1)
    if not os.path.exists(path):
        os.makedirs(path)
    path = folder.replace('out', 'eye', 1)
    if not os.path.exists(path):
        os.makedirs(path)
    path = folder.replace('out', 'landmark', 1)
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + '/landmarks.csv','w')
    csv_writer = csv.writer(f)
    for file in glob.glob(folder + '/*png'):
        face, eye, landmark = getRep(file)
        if face is None:
            continue
        cv2.imwrite(file.replace('out', 'face'), face)
        cv2.imwrite(file.replace('out', 'eye'), eye)
        PATHS.append([file.replace('out', 'face')] + landmark)
    csv_writer.writerows(PATHS)
    f.close()

modelDir = os.path.join('/Users/hayato/openface/models') # path要変更!!!!!!!
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

#black_face = np.zeros(96 * 96).reshape(96, 96, 1)
#black_eye = np.zeros(32 * 96).reshape(32, 96, 1)

def getRep(imgPath):
    """
    @param path
    return face image & eye image
    """
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)

    if bb is None:
        return None, None, None

    landmarks = align.findLandmarks(rgbImg, bb)
    landmarks_np = np.array(landmarks, dtype=np.float32)
    max_pt = np.max(landmarks_np, axis=0)
    min_pt = np.min(landmarks_np, axis=0)
    norm_landmarks = (landmarks_np - min_pt) / (max_pt - min_pt)  # 正規化
    norm_landmarks = norm_landmarks.reshape(136).tolist()
    alignedFace = align.align(96, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)  # 正規化,顔切り出し

    gray_image = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)  # グレースケール化
    hist_eq = cv2.equalizeHist(gray_image)  # ヒストグラム平坦化
    eye_image = hist_eq[:32, :]
    return hist_eq, eye_image, norm_landmarks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default = './out/yes/*',
                        help = 'Input directory of each people images( add * at the end because [glob] is used.)')
    args = parser.parse_args()

    print("#INPUT :",args.input)
    folder_list = sorted(glob.glob(args.input))
    assert len(folder_list) != 0 , 'don\'t find folder !!'
    for folder in folder_list:
        split_eyeimage(folder)
