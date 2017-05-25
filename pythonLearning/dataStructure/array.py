import array

symbols =  '$¢£¥€¤'
# The first argument of the array constructor defines the storage type 
# used for the numbers in the array

# not build a list. generate one element at a time.
arr = array.array('I', (ord(symbol) for symbol in symbols))
# build a list.
arr = array.array('I', [ord(symbol) for symbol in symbols])
print(arr[0])
