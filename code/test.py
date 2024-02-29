import numpy as np
coords = np.random.randint(0,3,size=(50,3))
l = 10
length = np.arange(l)

grid = np.meshgrid(length, length, length, indexing='ij')
sums = np.sum(grid, axis=0)
X, Y, Z = np.where(sums < l)

X_test, Y_test, Z_test = np.where(np.add.outer(np.add.outer(length, length), length) < l)

Q_exp = coords[..., np.newaxis]
powers = np.power(Q_exp, np.array([X, Y, Z]))  # Calculate powers for all coordinates

big_Q_exp = np.prod(powers, axis=1) 
b = 3