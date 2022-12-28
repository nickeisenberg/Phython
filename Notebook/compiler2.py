import numpy as np

x = np.array([1, 2, 33 ,4 ,5, 553])

dif = np.diff(x)

returns = dif / x[: -1]

print(dif)

print(returns)

print(x.std())

print(returns.std())

lr = np.log(x[1:] / x[:-1])

print(lr.std()) 

