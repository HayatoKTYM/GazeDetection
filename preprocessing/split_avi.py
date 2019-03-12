# -*- coding: utf-8 -*-

"""
AVI形式の動画ファイルを指定された開始時刻と
動画の時間を基に切り出して吐き出す
(出力はMP4形式推奨)
ただし，ffmpegをinstall する必要あり
"""
__author__ = "Hayato Katayama"
__date__    = "20190304"

import subprocess
import sys
import logging
import glob

def split_video(INPUT="", OUTPUT=""):
    if INPUT == "" or OUTPUT == "" or DURATION == "0.0":
        logging.exception("*** Cutting the video :: Argument setting incorrect ***")
    else:
        command = "ffmpeg -ss 0 -i " + INPUT +  " " + OUTPUT
        print("Command >> {}".format(command))
        subprocess.run(command, shell=True)


if __name__ == '__main__':
    files = glob.glob('*avi')#入力ファイルのdirec tory
    dir = ""#出力先のdirectory
    print(files)
    for file in files:
        output_file = dir + "image%4d" + str(i) + ".png"
        split_video(INPUT=movie_file, OUTPUT=output_file)
        print("Output >> {}".format(output_file))
