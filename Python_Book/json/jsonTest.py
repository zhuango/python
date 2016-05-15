import json

dumpStr = json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
print(dumpStr)

f = open('json', "w")
pathdict = {}
pathdict["corpus"] = "/home/corpus"
pathdict["intput"] = "/home/input"
path = json.dump(pathdict, f)
f.close()

f = open('json', 'r')
pathhook = json.load(f)
for key in pathhook:
    print(key + ":" + pathhook[key])