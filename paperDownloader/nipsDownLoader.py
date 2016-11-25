import urllib
from urllib2 import urlopen
from bs4 import BeautifulSoup
import os
import re

def FindURLs(website, label, labelAttrs=None):
    html = urlopen(website)
    soup = BeautifulSoup(html, "html.parser")
    papersNames = soup.findAll(label, attrs=labelAttrs)

    URLs = {}
    for item in papersNames:
        try:
            paperName = list(item.children)[0].text
            href      = list(item.children)[0].attrs['href']
            if href.startswith("http://"):
                paperAddr = href
            else:
                paperAddr = "https://papers.nips.cc/" + href + ".pdf"
            URLs[paperName] = paperAddr
        except:
            continue
    return URLs
# 2015:https://papers.nips.cc/book/advances-in-neural-information-processing-systems-28-2015
# 2014:https://papers.nips.cc/book/advances-in-neural-information-processing-systems-27-2014
# 2016:https://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016
website = "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016"
rootPath = "/home/laboratory/Documents/ebook/papers/NIPS/2016/"

if not os.path.exists(rootPath):
    os.makedirs(rootPath)
URLs = FindURLs(website, 'li')
counter = 0
print(len(URLs))
for name in URLs:
    counter += 1
    print("downloading " + name)
    path = rootPath + name.replace("/", " ") + ".pdf"
    if os.path.exists(path):
        continue
    try:
        urllib.urlretrieve(URLs[name], path)
        print(path)
    except KeyboardInterrupt:
        break
    except:
        continue
print ("Total: " + str(counter))