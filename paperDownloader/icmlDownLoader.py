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

# 2016: http://jmlr.org/proceedings/papers/v48/
# 2015: http://jmlr.org/proceedings/papers/v37/
# 2014: http://jmlr.org/proceedings/papers/v32/
website = "http://jmlr.org/proceedings/papers/v32/"
rootPath = "/home/laboratory/Documents/ebook/papers/ICML/2014/"

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
    urllib.urlretrieve(URLs[name], path)
print ("Total: " + str(counter))