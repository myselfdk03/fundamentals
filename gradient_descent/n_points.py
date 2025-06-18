import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
n = len(x)
m = 0
c = 0
lr = 0.01
epoch = 100
y_pred = m*x + c

def dL_dm(y, y_pred):
  return (-2/n) * np.sum((y - (y_pred))*x)

def dL_dc(y, y_pred):
  return (-2/n) * np.sum(y - y_pred)

for i in range(epoch):
  y_pred = m*x + c
  m = m - lr*dL_dm(y, y_pred)
  c = c - lr*dL_dc(y, y_pred)
  print(m, c)

