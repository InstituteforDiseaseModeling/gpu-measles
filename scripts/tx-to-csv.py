#!/usr/bin/env python3

import numpy as np

data = np.fromfile('transmission.bin', np.uint16)
rows = data.shape[0] // 3
data = data.reshape((3,rows))
data = data.transpose()
np.savetxt('transmission.csv', data, delimiter=',')
