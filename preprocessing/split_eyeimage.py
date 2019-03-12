__author__ = "Hayato Katayama"
__date__    = "20190304"

import cv2, os
import subprocess
import openface
import numpy as np

def split_eyeimage(movie_file):
    """
    動画を読み込んで画像を切り出す(一応保存)
    画像から顔を検出して切り出す(一応保存)
    目の部分を切り取る(要保存)

    @param: movie_file (.mp4)
    動画(mp4)を画像(png)に変換したものを保存する
    同directory内のfaceフォルダに顔画像を保存する(96*96)
    同directory内のeyeフォルダに顔画像を保存する(96*32)

    """
    #movie -> image
    folder = split_png(movie_file)
    #image -> face & eye image
    path = folder.replace('img', 'face', 1)
    if not os.path.exists(path):
        os.mkdir(path)
    path = folder.replace('img', 'eye', 1)
    if not os.path.exists(path):
        os.mkdir(path)
    for file in glob.glob(folder + '/*png'):
        face, eye = getRep(file)
        cv2.imwrite(file.replace('img', 'face'), face)
        cv2.imwrite(file.replace('img', 'eye'), eye)


# movie >> image
def split_png(file):
    """
    @param movie path
    return save path
    """
    folder = "/Users/hayato/Desktop/img/" + file.split("/")[-1].split(".")[0]
    if not os.path.exists(folder):
        os.mkdir(folder)

    command = "ffmpeg -i " + file + " -ss 0 -r 2 -f image2 " + folder + "/%05d.png"  # 0秒からフレームレート10でpngとして取り出す
    subprocess.run(command, shell=True)
    print(command)
    return folder


modelDir = os.path.join('/Users/hayato/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

black_face = np.zeros(96 * 96).reshape(96, 96, 1)
black_eye = np.zeros(32 * 96).reshape(32, 96, 1)


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

    # if no face -> return black image
    if bb is None:
        return black_face, black_eye

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
    return hist_eq, eye_image
