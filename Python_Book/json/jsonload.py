import json


f = open('test.json', 'r')
pathhook = json.load(f)
for key in pathhook:
    print(key + ":" + pathhook[key])