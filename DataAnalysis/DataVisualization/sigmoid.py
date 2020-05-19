import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1./(1 + np.e**(-x))

# Calculate plot points
x = np.arange(-5., 5., 0.01)
y = sigmoid(x)
dx = y*(1 - y)

# Setup centered axes
fig, ax = plt.subplots(figsize=(10, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Create and show plot
ax.plot(x, y, color='orange', linewidth=3, label='sigmoid')
ax.plot(x, dx, color='g', linewidth=3, label='derivative')
plt.show()
