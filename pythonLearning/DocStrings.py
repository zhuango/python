def myfunction():
    """Do nothing, document it.
        
    No, really, it does nothing.
    """
    pass
    
print(myfunction.__doc__)

import os 
print(os.path.dirname("/asd/ads/test.xtt"))

#os.path.mkdir

cat = 0
print('sdfs', 'asdf', 10, 123, ')')

dim_proj = 150
category = 'book'
test_err = 0.0000000123001023
train_err = 0.12312312321
eidx = 12
uidx = 1000
cost = 0.123123123
cost1 = 0.123123123
cost2 = 0.23423423
diffStr = " (" + category + " " +str(dim_proj) + "d)"
print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Cost1', cost1, 'Cost2', cost2, diffStr
print('11111111111' + diffStr)
print ('Train ', train_err, 'Test ', test_err, diffStr)

print 'The code run for %d epochs, with %f sec/epochs' % (
    (eidx + 1), (23 - 1) / (1. * (eidx + 1))), diffStr
fint = float('inf')
print(fint)

def test(l):
    l[0] = 1000
ls = [[10,20, 30], [40, 50,60]]
first = list(ls[0])
test(first)
print(ls)

for i in range(10): 
    print(i)
    print(i)