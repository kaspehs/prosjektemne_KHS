import numpy as np
import matplotlib.pyplot as plt
m = 16.79
k = 1218.0
c = 2.86
ma = 1.0*0.25*0.1**2*np.pi*1000
m += ma

data = np.load("data.npz")
print(data['c'])
t = data["a"].astype(np.float32)      # [T]
y = data["b"].astype(np.float32)      # [T]
F_true_available = "c" in data
F_true = data["c"].astype(np.float32) if F_true_available else None

dy = np.zeros_like(y, dtype=np.float32)
ddy = np.zeros_like(y, dtype=np.float32)

# First derivative (central, one-sided at edges)
dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
dy[0]    = (y[1] - y[0]) / (t[1] - t[0])
dy[-1]   = (y[-1] - y[-2]) / (t[-1] - t[-2])

# Second derivative (central using dy; more stable than direct 2nd diff on y)
ddy[1:-1] = (dy[2:] - dy[:-2]) / (t[2:] - t[:-2])
ddy[0]    = (dy[1] - dy[0]) / (t[1] - t[0])
ddy[-1]   = (dy[-1] - dy[-2]) / (t[-1] - t[-2])

F_ext = ddy*m + dy*c + y*k

if F_true is not None:
    plt.plot(t, F_true, label='Data Force')
plt.plot(t, F_ext, label = 'extrapolated force')
plt.legend()
plt.show()
