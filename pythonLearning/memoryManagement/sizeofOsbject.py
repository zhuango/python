#!/usr/bin/python2

import sys
# def show_sizeof(x, level=0):
#     print("{0} {1} {2} {3}".format("\t" * level,str(x.__class__),str(sys.getsizeof(x)),str(x)))
#     if hasattr(x, '__iter__'):
#         if hasattr(x, 'items'):
#             for xx in x.items():
#                 show_sizeof(xx, level + 1)
#         else:
#             for xx in x:
#                 show_sizeof(xx, level + 1)

def show_sizeof(x, level=0):

    print("{0} {1} {2} {3}".format("\t" * level,str(x.__class__),str(sys.getsizeof(x)),str(x)))

    if hasattr(x, '__iter__'):
        if hasattr(x, 'items'):
            for xx in x.items():
                show_sizeof(xx, level + 1)
        else:
            for xx in x:
                show_sizeof(xx, level + 1)
                
show_sizeof(None)
show_sizeof(3)
show_sizeof(2**63)
show_sizeof(102947298469128649161972364837164)
show_sizeof(918659326943756134897561304875610348756384756193485761304875613948576297485698417)
show_sizeof(3.14159265358979323846264338327950288)
show_sizeof("")
show_sizeof("My hovercraft is full of eels")
show_sizeof([])
show_sizeof([4, "toaster", 230.1])
show_sizeof({})
show_sizeof({'a':213, 'b':2131})