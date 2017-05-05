#!/usr/bin/python3

import torch
from torch.autograd import Variable

x = torch.randn((2, 2))
y = torch.randn((2, 2))
z = x + y

var_x = Variable(x)
var_y = Variable(y)

var_z= var_x + var_y
print(var_z.creator)

var_z_data = var_z.data
new_var_z = Variable(var_z.data)

# ... does new_var_z have information to backprop to x and y?
# NO!
print(new_var_z.creator)
# And how could it?  We yanked the tensor out of var_z (that is
# what var_z.data is).  This tensor doesn't know anything about
# how it was computed.  We pass it into new_var_z, and this is all the
# information new_var_z gets.  If var_z_data doesn't know how it was
# computed, theres no way new_var_z will.
# In essence, we have broken the variable away from its past history


# If you want the error from your loss function to backpropogate to a
# component of your network, you MUST NOT break the Variable chain from 
# that component to your loss Variable. If you do, the loss will have no idea 
# your component exists, and its parameters can’t be updated.

newView = var_z.view(4,1)
print(newView.creator)