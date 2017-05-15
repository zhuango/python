#!/usr/bin/python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("echo")
parser.add_argument("square", help='display a square of a given number.',
                    type=int)
args = parser.parse_args()
print(args.echo)
print(args.square**2)
# python3 positionalArguments.py echo 12