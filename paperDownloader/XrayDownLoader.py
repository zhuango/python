#!/usr/bin/python

import urllib
#from urllib3 import urlopen
import requests
import json
import time
import path
import os

def downLoadImage(req, imagePath):
    html = ""
    try:
        html = urllib.request.urlopen(req, timeout=10).read()
    except:
        return False
    temp_file = open(imagePath, 'wb')
    temp_file.write(html)
    temp_file.close()
    return True

datasetPath = "./x_ray/"
imagePath = datasetPath + "images/"

fromGithub = True
count = 7471
i = 1
imageId = 1
while i < count:
    url = "https://openi.nlm.nih.gov/retrieve.php?query=Indiana%20University%20Chest%20X-ray%20Collection&m={}&n={}".format(i, min([i + 99, count]))
    r = requests.get(url)
    strings = r.content.decode("utf-8")
    content = json.loads(strings)
    
    for instance in content["list"]:
        try:
            ########## image ##########
            if imageId % 5 == 0:
                fromGithub = not fromGithub

            # github or openi
            if fromGithub:
                imageUrl = "https://github.com/xinyuaaanz/Image-Captioning-for-chest-x-ray/raw/master/preprocessing/images/" + instance["imgLarge"].split("/")[-1]
            else:
                imageUrl = "https://openi.nlm.nih.gov" + instance["imgLarge"]

            imageFullName = "{}{}".format(imagePath, instance["imgLarge"].split("/")[-1])
            if not os.path.exists(imageFullName):
                # When fail on downloading image, try it at most 3 times.
                retry = 3
                while retry > 0:
                    print("downloading {}......".format(imageFullName))
                    succe = downLoadImage(imageUrl, imageFullName)
                    # Do not bother a website too much and sleep for a while.
                    time.sleep(3)
                    if succe:
                        break
                    retry -= 1
            else:
                print("{} is already downloaded".format(imageFullName))
                
            ########## report ##########
            #impressionFullName = "{}{}.json".format(datasetPath, str(imageId))            
            #f = open(impressionFullName, "w")
            #instance["imagePath"] = imageFullName
            #json.dump(instance, f)
            #f.close()

            imageId += 1
        except :
            print("you are dead!!!")
            time.sleep(60 * 10)
    i += 100
    print(i)

# url = "https://openi.nlm.nih.gov/detailedresult?img=CXR1921_IM-0598-1001&query=Indiana%20chest%20X-ray%20collection&it=xg&req=4&npos=85"
# r = requests.get(url)
# strings = r.content.decode("utf-8")
# print(strings)