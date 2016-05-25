while True:
    try:
        x = int(input("Please enter a number: "))
        print(x)
        break
    except ValueError:
        print("Oops! That was no valid number. Try again...")

import sys

try:
    f = open('myfile.txt')
    s = f.readline()
    i = int(s.strip())
except OSError as err:
    print("OS error: {0}".format(err))
except (ValueError):
    print("Could not convert data to an teger.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise # throw
    
for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except IOError:
        print("cannot open", arg)
    else: # if the try clause does not raise an exception.
        print(arg, 'has', len(f.readline()), 'lines')
        f.close()
        
try:
    raise Exception('spam', 'eggs')
except Exception as inst:
    print(type(inst))
    print(inst.args)
    print(inst)
    
    x, y = inst.args
    print('x = ', x)
    print('y = ', y)
    
def this_fails():
    x = 1 / 0
try:
    this_fails()
except ZeroDivisionError as err:
    print("handling run-time error:", err)