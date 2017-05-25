#!/usr/bin/python3

# evaluate the expression seq[start:stop:step], Python calls
# seq.__getitem__(slice(start, stop, step))

invoice = """
0.....6.................................40........52...55........
1909 Pimoroni PiBrella $17.50 3 $52.50
1489 6mm Tactile Switch x20 $4.95 2 $9.90
1510 Panavise Jr. - PV-201 $28.00 1 $28.00
1601 PiTFT Mini Kit 320x240 $34.95 1 $34.95
"""
SKU = slice(0, 6)
DESCRIPTION= slice(6, 40)
UNIT_PRICE = slice(40, 52)
ITEM_TOTAL = slice(52, 55)
line_items = invoice.split("\n")[2:]
for item in line_items:
    print(item[UNIT_PRICE], item[DESCRIPTION])

# evaluate a[i, j], Python calls a.__getitem__((i, j)).
# ...is an alias to the Ellipsis object, the single
# instance of the ellipsis class

# if x is a 4- dimensional array, x[i, ...] is a shortcut for x[i, :, :, :,]

print("______assign to a slice_________")
l = list(range(10))
l[2:5] = [20, 30]
print(l)

del l[5:7]
print(l)

l[3::2] = [11, 22]
print(l)

# error! 
# When the target of the assignment is a slice, the right-hand side must be an
# iterable object, even if it has just one item.
# l[2:5] = 100
l[2:5] = [100]
print(l)