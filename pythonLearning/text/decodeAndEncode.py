filename = "/home/laboratory/github/leetCode/情感分析60000/pos60000_copy.txt"
fw = open(filename +"utf-8", 'wb')
with open(filename, 'rb') as f:
    for line in f:
        # source encoding
        line = line.decode("gbk")
        # text
        # target encoding
        line = line.encode("utf-8")
        fw.write(line)
fw.close()