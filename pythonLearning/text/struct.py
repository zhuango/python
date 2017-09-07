import struct
# < little-endian; 3s3s two sequences of 3 bytes; HH two 16-bit integers.
fmt = '<3s3sHH'
with open("filter.gif", 'rb') as fp:
    img = memoryview(fp.read())
header= img[:10]
print(bytes(header))
print(struct.unpack(fmt, header))
del header
del img


