import pylab as pl
import numpy as np


x = np.arange(-10, 10, 0.1)

f1 = (x-1)**2
f2 = np.exp(-(x-1)**2)*50


pl.plot(x, f1, label='f1: Behavioral feature 1')
pl.plot(x, f2, label='f2: Behavioral feature 2')
pl.xlabel("x")
pl.ylabel("Error function value")
pl.legend()
pl.ylim(0, 120)
pl.savefig("Example1.png")

pl.figure()
f2 = np.exp(-(np.sqrt(f1)+1-1)**2)*50
pl.plot(f1, f2, label='Pareto front')
pl.xlabel("f1")
pl.ylabel("f2")
pl.legend()

pl.xlim(0, 20)
pl.ylim(0, 20)
pl.savefig("Example_pareto.png")
