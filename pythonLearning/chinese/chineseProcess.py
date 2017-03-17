#/usr/bin/python3

outputFile = open("chineseProcess_output.txt", 'w')
with open("chineseProcess_input.txt", 'r') as f:
    for line in f:
        print(line[0])
        print(line[1])
        outputFile.write(line)