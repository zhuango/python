import argparse
import numpy

lengthOfFragment = 5
def generate(wordslist, vectorsDict, dimension):
    parser = argparse.ArgumentParser()
    parser.add_argument('--vo', default=wordslist, type=str)
    parser.add_argument('--ve', default=vectorsDict, type=str)
    parser.add_argument('--di', default=dimension, type=str)
    args = parser.parse_args()
    
    wordslist = args.vo
    vectorsDict = args.ve
    dimension = int(args.di)
    # with open(args.vo, 'r') as f:
    #     for line in f:
    #         for word in line.rstrip().split(' ')
    #             words.append(word)
    # words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    ####################################
    linenumber = 0
    ####################################
        
    with open(vectorsDict, 'r') as f:
        vectors = {}
        for line in f:
            linenumber += 1   #
            if (linenumber % 1000 == 0): print(linenumber) #
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
    ####################################
    linenumber = 0
    notFoundCount = 0
    ####################################
    
    vecfile = open(wordslist +"_" + str(dimension) + ".vector", "w")
    with open(wordslist, 'r') as f:
        for line in f:
            fragmentVector=""
            linenumber += 1  # ################################
            if (linenumber % 1000 == 0): print(linenumber)# 
            for word in line.rstrip().split(' '):
                try:
                    for elem in vectors[word]:
                        fragmentVector = fragmentVector + str(elem) + " "
                except Exception:
                    notFoundCount += 1 ###
                    # print(word)
                    # random
                    for elem in numpy.random.rand(dimension) - 0.5:
                        fragmentVector = fragmentVector + str(elem) + " "
            vecfile.writelines(fragmentVector + "\n")
    print("There are " + str(notFoundCount) + " words witch are not found.")
if __name__ == "__main__":
    vectorsDict = "C:\Users\liuz\Desktop\cnn\cn\glove.twitter.27B.50d.txt"
    wordslist = "C:\Users\liuz\Desktop\cnn\cn\label_book_new.txt.extract"
    generate("", "", 25)