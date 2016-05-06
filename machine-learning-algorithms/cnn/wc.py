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
    
if __name__ == "__main__":
    wc("")