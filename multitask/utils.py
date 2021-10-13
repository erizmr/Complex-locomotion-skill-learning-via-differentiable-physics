import time
import sys

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import taichi as ti

real = ti.f64
scalar = lambda: ti.field(dtype=real)
vec = lambda dim: ti.Vector.field(dim, dtype=real)
mat = lambda dim: ti.Matrix.field(dim, dim, dtype=real)

class Debug():
    def __init__(self, debug = True):
        self.debug = debug
        self.pref = 0
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            print("{}# {} start...".format(' ' * self.pref, func.__name__))
            self.pref += 4

            sys.stdout.flush()

            ret = func(*args, **kwargs)

            t2 = time.time()
            self.pref -= 4
            print("{}# {} end in {}s".format(' ' * self.pref, func.__name__, t2 - t1))
            sys.stdout.flush()
            return ret
        if self.debug:
            return wrapper
        return func

if __name__ == "__main__":
    debug = Debug(True)
    @debug
    def run(t):
        if t == 0:
            q = 0
            for i in range(10000000):
                q = q + 1
        else:
            run(t - 1)
            q = 0
            for i in range(10000000):
                q = q + 1
    
    run(5)

def plot_curve(plots, path = "output.png"):
    fig = plt.figure()
    plt.plot(plots, color = "red")
    temp_curve = gaussian_filter(plots, 10)
    plt.plot(temp_curve)
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)

def load_string(message, options = ["Y", "N"]):
    print(message)
    s = input()
    while s not in options:
        print(message)
        s = input()
    return s