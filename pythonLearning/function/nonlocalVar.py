def make_averager():
    count = 0
    total = 0
    
    def averager(new_value):
        # count is inmutable variable. Thus count += 1 equals count = count + 1
        count += 1
        total += new_value
        return total / count
    return averager
ave = make_averager()
#print(ave(10))
#print(ave(11))

def make_averager():
    count = 0
    total = 0
    def averager(new_value):
        nonlocal count
        nonlocal total
        count += 1
        total += new_value
        return total / count
    return averager
ave = make_averager()
print(ave(10))
print(ave(11))

