#!/usr/bin/python
def comp7Sort(array):
    if not len(array) == 5:
        return None

    a, b, c, d, e = array[0], array[1], array[2], array[3], array[4]
    
    if b > a:
        a, b = b, a
    if d > c:
        c, d = d, c
    # a > b, c > d

    if c > a:
        a, c = c, a
    # a > b, a > c > d

    if e > c:
        if b > c: # a > b > c > d, e > c > d
            if   e > a: # e > a > b > c > d
                array[0], array[1], array[2], array[3], array[4] = d, c, b, a, e
                print("1") #debug
            elif e > b: # a > e > b > c > d
                array[0], array[1], array[2], array[3], array[4] = d, c, b, e, a
                print("2") #debug
            else:       # a > b > e > c > d
                array[0], array[1], array[2], array[3], array[4] = d, c, e, b, a
                print("3") #debug
        else: # a > c > b, e > c > d
            if e > a: # e > a > c > b, c > d, (b ? d)
                array[2], array[3], array[4] = c, a, e
                print("4") #debug
            else:     # a > e > c > b, c > d, (b ? d)
                array[2], array[3], array[4] = c, e, a
                print("5") #debug
            
            if b > d:# b > d
                array[0], array[1] = d, b
                print("_1") #debug
            else:    # b < d
                array[0], array[1] = b, d
                print("_2") #debug
    else:
        if e > d: # a > c > e > d, a > b
            if   b < e: 
                if b < d: # a > c > e > d > b
                    array[0], array[1], array[2], array[3], array[4] = b, d, e, c, a
                    print("6") #debug
                else:     # a > c > e > b > d
                    array[0], array[1], array[2], array[3], array[4] = d, b, e, c, a
                    print("7") #debug
            else:
                if b < c: # a > c > b > e > d
                    array[0], array[1], array[2], array[3], array[4] = d, e, b, c, a
                    print("8")
                else:     # a > b > c > e > d
                    array[0], array[1], array[2], array[3], array[4] = d, e, c, b, a
                    print("9") #debug
        else: # a > c > d > e, a > b
            if b < d:
                if b < e: # a > c > d > e > b
                    array[0], array[1], array[2], array[3], array[4] = b, e, d, c, a
                    print("10") #debug
                else:     # a > c > d > e > b
                    array[0], array[1], array[2], array[3], array[4] = e, b, d, c, a
                    print("11") #debug
            else:
                if b < c: # a > c > b > d > e
                    array[0], array[1], array[2], array[3], array[4] = e, d, b, c, a
                    print("12") #debug
                else:     # a > b > c > d > e
                    array[0], array[1], array[2], array[3], array[4] = e, d, c, b, a
                    print("13") #debug
    
    return array

result = comp7Sort([4, 3, 2, 1, 5])
print(result)

result = comp7Sort([5, 3, 2, 1, 4])
print(result)

result = comp7Sort([5, 4, 2, 1, 3])
print(result)

result = comp7Sort([4, 2, 3, 1, 5])
print(result)

result = comp7Sort([4, 1, 3, 2, 5])
print(result)

result = comp7Sort([5, 2, 3, 1, 4])
print(result)

result = comp7Sort([5, 1, 3, 2, 4])
print(result)

result = comp7Sort([5, 1, 4, 2, 3])
print(result)

result = comp7Sort([5, 2, 4, 1, 3])
print(result)

result = comp7Sort([5, 3, 4, 1, 2])
print(result)

result = comp7Sort([5, 4, 3, 1, 2])
print(result)

result = comp7Sort([5, 1, 4, 3, 2])
print(result)

result = comp7Sort([5, 2, 4, 3, 1])
print(result)

result = comp7Sort([5, 3, 4, 2, 1])
print(result)

result = comp7Sort([5, 4, 3, 2, 1])
print(result)
