import numpy as np
from mpl_toolkits.mplot3d import Axes3D #line 18 has use it
from matplotlib import pyplot as plt
import torch


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(40, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# [1., 0.], [-4, 0.], [4, 0.]
x = torch.tensor([-4., 0.], requires_grad=True)

# no loss func, we optimize x, the predict value.
#   optimizing object is not weights, but x.
# what adam optimizer do is :   x' = x - 0.001*delta(x)
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):

    pred = himmelblau(x)
    # clear gradient information, as backward function
    #   will accumulate grad value.
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print('step {}: x = {}, f(x) = {}'
              .format(step, x.tolist(), pred.item()))
        # at last, predict stays at minimal value:0, pred = himmmelblau(x)
