#/usr/bin/python3
from collections import defaultdict

def genSentimentWords():
    newSentimentWords = "newSentimentWords.txt"
    newSentiWordsStream = open(newSentimentWords, 'w')
    with open("sentimentwords.txt", "r") as f:
        for line in f:
            items = line.strip().split(" ")
            por = items[6]
            if   por == '1':
                newSentiWordsStream.write(items[0] + " " + "1\n")
            elif por == '2':
                newSentiWordsStream.write(items[0] + " " + "0\n")
    newSentiWordsStream.close()

def statisticSentiword():
    newSentimentWords = "newSentimentWords.txt"
    fileList = ["/home/laboratory/corpus/denoise/cn/label_book_new.txt",
                "/home/laboratory/corpus/denoise/cn/label_dvd_new.txt",
                "/home/laboratory/corpus/denoise/cn/label_music_new.txt",
                "/home/laboratory/corpus/denoise/cn/test_book_new.txt",
                "/home/laboratory/corpus/denoise/cn/test_dvd_new.txt",
                "/home/laboratory/corpus/denoise/cn/test_music_new.txt"]
    sentimentWordsDict = defaultdict(lambda:-1)
    sentimentwordsPorl = defaultdict(lambda:'0')
    with open(newSentimentWords, 'r') as f:
        for line in f:
            items = line.strip().split(" ")
            sentimentWordsDict[items[0]] = 0
            sentimentwordsPorl[items[0]] = items[1]
    for file in fileList:
        with open(file, 'r') as f:
            for line in f:
                for word in line.strip().split(" "):
                    word = word.strip()
                    if  sentimentWordsDict[word] >= 0:
                        sentimentWordsDict[word] += 1
    with open("statistic.txt", 'w') as f:
        for item in sorted(sentimentWordsDict.items(), key=lambda d:d[1], reverse=True):
            if sentimentWordsDict[item[0]] > 0:
                f.write(item[0] + " " +str(sentimentwordsPorl[item[0]]) + " " + str(sentimentWordsDict[item[0]]) + '\n')
            #f.write(item + " " + str(sentimentWordsDict[item]) + '\n')
statisticSentiword()