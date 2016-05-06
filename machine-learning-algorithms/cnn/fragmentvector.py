import argparse

vectorsDict = "/home/laboratory/corpus/glove.twitter.27B.50d.txt"
wordslist = "/home/laboratory/corpus/train_en_0122/label_music_new.txt.extract"

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default=wordslist, type=str)
    parser.add_argument('--vectors_file', default=vectorsDict, type=str)
    args = parser.parse_args()

    # with open(args.vocab_file, 'r') as f:
    #     for line in f:
    #         for word in line.rstrip().split(' ')
    #             words.append(word)
    # words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    ####################################
    linenumber = 0
    ####################################
        
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            linenumber += 1
            print(linenumber)
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
    ####################################
    linenumber = 0
    ####################################
    
    vecfile = open(vectorsDict + ".vector", "w")
    with open(args.vocab_file, 'r') as f:
        for line in f:
            fragmentVector=""
            linenumber += 1 # ################################
            print(linenumber)
            for word in line.rstrip().split(' '):
                try:
                    for elem in vectors[word]:
                        fragmentVector = fragmentVector + str(elem) + " "
                except Exception:
                    print(Exception.args)
            vecfile.writelines(fragmentVector + "\n")
            
if __name__ == "__main__":
    generate()  