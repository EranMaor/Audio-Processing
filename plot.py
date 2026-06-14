from enum import auto
import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

def plot(x, y, title, xlabel, ylabel, xlim):
    plt1.figure(figsize=(10, 4))
    plt1.title(title)
    plt1.ylabel(ylabel)
    plt1.xlabel(xlabel)
    plt1.xlim(left=xlim[0], right=xlim[1])
    plt1.plot(x, y)
    plt1.tight_layout()
    plt1.show()

    plt2.figure(figsize=(10, 4))
    plt1.title('title')
    plt2.show()