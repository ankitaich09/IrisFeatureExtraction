# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:00:54 2019

@author: cisguest
"""
from sys import argv

def add(a,b):
    return sum(a,b)

def main():
    print(argv)
    file, a,b = argv
    a = int(a)
    b = int(b)
    c = a+b
    print(c)
    
if __name__ == "__main__":
    main()