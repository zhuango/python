#python .\genWordVector.py --vo "G:\liuzhuang\corpus\cn\label_dvd_new.txt.extract" --ve "G:\liuzhuang\corpus\cn_vectorTable\cn_vectors_50.txt" --di 50
import os
import theano
import numpy

print(len("0.0895 -0.0931 -0.0975 -0.0786 0.1019 -0.1304 0.2608 0.0622 0.0390 -0.0292 -0.0974 0.0482 -0.0151 0.1271 -0.2610 -0.0738 0.0047 -0.0635 0.0442 0.0237 -0.0180 0.1093 0.0167 -0.3237 0.0558 0.0398 -0.0089 0.2941 -0.2788 -0.0622 0.0450 0.0665 0.2023 0.0849 -0.1633 0.0240 0.1719 0.0661 -0.1919 0.0855 -0.1645 0.0583 -0.0073 -0.0985 -0.1600 0.1864 -0.1390 0.0007 -0.0855 -0.0004".split(" ")))

testPath = "G:/liuzhuang/MUHAHA"
os.makedirs(testPath)
print(os.path.exists(testPath))