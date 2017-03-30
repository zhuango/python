from collections import OrderedDict
import copy
import gzip
import os
import urllib
import random
import stat
import subprocess
import sys
import timeit

import numpy

import theano
from theano import tensor as T
import subprocess

subprocess.call(["mkdir", 'test'])
subprocess.call(['./print'])