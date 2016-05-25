import json
# serialization
print(json.dumps([1, 'simple', 'list']))

print("#####################")
f = open("json", "w")
x = [1, 'simple', 'list']
json.dump(x, f)
f.close()

f = open("json", 'r')
y = json.load(f)
print(y)
f.close()