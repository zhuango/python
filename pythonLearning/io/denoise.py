#/usr/bin/python3

stopWords = ['啊', '的', '了', '和', '是', '就', '都', '而', '及']
symbols = ['≦','▽', '≧', '\\', '＊','ω', '↖', '↗', '；', ':', '?', '_','^', '…', '～', '`', '#', '&', '、',')', '(','!',',' ,'~', '.', '，', '', '-','=', '%', '！', '/', '：', '？', '*', ';', '·' , '+' , '”', '“', '（', '）', '\'', '。','％', '＃', '$','＆', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',]
def denoiseCN(filename, newFilename):
    fileStream      = open(filename, 'r')
    newFileStream   = open(newFilename, 'w')
    for line in fileStream:
        if line.startswith('<') and line.endswith('>\n'):
            newFileStream.write(line)
            continue
        line = line.lower()
        for word in stopWords:
            line = line.replace(" " + word + " ", " ")
        for symbol in symbols:
            line = line.replace(symbol, "")
        if not line == '\n':
            sentence = ""
            for word in line.strip().split(" "):
                if word.startswith(" ") or word == "":
                    continue
                sentence += word.strip() + " "
            if not sentence == '\n':
                newFileStream.write(sentence.strip() + "\n")
def denoiseEN(filename, newFilename):
    fileStream      = open(filename, 'r')
    newFileStream   = open(newFilename, 'w')
    for line in fileStream:
        if line.startswith('<') and line.endswith('>\n'):
            newFileStream.write(line)
            continue
        line = line.lower()
        for word in stopWords:
            line = line.replace(" " + word + " ", " ")
        for symbol in symbols:
            line = line.replace(symbol, "")
        if not line == '\n':
            sentence = ""
            for word in line.strip().split(" "):
                if word.startswith(" ") or word == "":
                    continue
                sentence += word.strip() + " "
            if not sentence == '\n':
                newFileStream.write(sentence.strip() + "\n")

root = "/home/laboratory/corpus/denoise/cn/"
denoiseCN(root + "label_music_new.txt", root + "processed/label_music_new.txt")