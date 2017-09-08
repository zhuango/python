s = 'café'
print(len(s))

b = s.encode('utf8')
print(b)
print(len(b))

b_decoded = b.decode('utf8')
print(b_decoded)

for codec in ['latin_1', 'utf_8', 'utf_16']:
    print(codec, 'El Niño'.encode(codec), sep='\t')

