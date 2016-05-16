import argparse

def wc(fileName):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default=fileName, type=str)
    args = parser.parse_args()
    fileName = args.f
    # fileName = "C:\Users\liuz\Desktop\corpus\label_book_new.txt.vector"
    lineNumber = 0
    with open(fileName, "r") as f:
        for line in f:
            if line != "":
                lineNumber += 1

    print(fileName + " count: "+ str(lineNumber))
    return lineNumber

def wcWord(fileName):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default=fileName, type=str)
    args = parser.parse_args()
    fileName = args.f
    # fileName = "C:\Users\liuz\Desktop\corpus\label_book_new.txt.vector"
    wordNumber = 0
    with open(fileName, "r") as f:
        for line in f:
            if line != "":
                for word in line.strip().split(" "):
                    wordNumber += 1

    print(fileName + " word count: "+ str(wordNumber))
    return wordNumber

def maxWordLen(fileName):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default=fileName, type=str)
    args = parser.parse_args()
    fileName = args.f
    # fileName = "C:\Users\liuz\Desktop\corpus\label_book_new.txt.vector"
    maxLen = 0
    with open(fileName, "r") as f:
        for line in f:
            lineLen = len(line.strip().split(" "));
            if( lineLen> maxLen):
                maxLen = lineLen

    print(fileName + " max Length: "+ str(maxLen))
    return maxLen
def IsContentSame(filenameA, filenameB):
    a = open(filenameA, "r")
    b = open(filenameB, "r")
    linenumber = 0
    for linea in a:
        linenumber+=1
        lineb = b.readline().strip()
        if(not (linea.strip() == lineb)): 
            print(linenumber)
            print(">"+linea.strip() +"$")
            print("<"+lineb +"$")
            return False
    return True
    
if __name__ == "__main__":
    # filenames = []
    # filenames.append("G:/liuzhuang/corpus/Serializer/semantic_sentiment_test_music_en_50.txt")
    # filenames.append("G:/liuzhuang/corpus/Serializer/semantic_sentiment_train_music_cn_50.txt")
    # filenames.append("G:/liuzhuang/corpus/Serializer/semantic_sentiment_train_music_en_50.txt")
    # filenames.append("G:/liuzhuang/corpus/Serializer/semantic_sentiment_test_music_cn_50.txt")
    # wordsnumber=0
    # maxLen = 0
    # for filename in filenames:
    #     wordsnumber += wcWord(filename)        
    #     tmpLen = maxWordLen(filename)
    #     if(maxLen < maxWordLen(filename)):
    #         maxLen = tmpLen
    # print(wordsnumber)
    # print(maxLen)
    result = IsContentSame("cpp/test_book_new.txt.extract_50.serialization", "cpp/test_book_new.txt.extract_50cn.serialization")
    print(result)