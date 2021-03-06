import urllib
from urllib2 import urlopen
from bs4 import BeautifulSoup
import os
import re

def FindURLs(website, label, labelAttrs):
    html = urlopen(website)
    soup = BeautifulSoup(html, "html.parser")
    papersNames = soup.findAll(label, attrs=labelAttrs)

    URLs = {}
    for item in papersNames:
        paperName = list(item.children)[1].text
        href = list(list(item.children)[5].children)[3].attrs['href']
        if href.startswith("http://"):
            paperAddr = href
        else:
            paperAddr = website + "/" + href
        URLs[paperName] = paperAddr
    return URLs

# 2017: http://proceedings.mlr.press/v70/
# 2016: http://jmlr.org/proceedings/papers/v48/
# 2015: http://jmlr.org/proceedings/papers/v37/
# 2014: http://jmlr.org/proceedings/papers/v32/
# colt 2017: http://proceedings.mlr.press/v65/
website = "http://proceedings.mlr.press/v65/"
rootPath = "/home/laboratory/Documents/ebook/papers/COLT/2017/"

if not os.path.exists(rootPath):
    os.makedirs(rootPath)
URLs = FindURLs(website, 'div', {"class":"paper"})
counter = 0
for name in URLs:
    counter += 1
    print("downloading " + name)
    path = rootPath + name.replace("/", " ") + ".pdf"
    if os.path.exists(path):
        continue
    try:
        urllib.urlretrieve(URLs[name], path)
    except KeyboardInterrupt:
        break
    except:
        continue
print ("Total: " + str(counter))