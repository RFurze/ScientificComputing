import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def plot_solution_3d( sol, h ):
  n = int(1./h + 1)
  X = np.arange(0., 1.+h, h)
  Y = np.arange(0., 1.+h, h)
  X, Y = np.meshgrid(X, Y)
  Z = sol.reshape(n,n)

  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  ax.set_zlim(0.0, 0.01)
  fig.tight_layout(pad=1.0)
  plt.xlabel('x')
  plt.ylabel('y')
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patches as patches

def plot_solution_2d(solgrad,h):

  fig, ax = plt.subplots()

  # Make data.
  X = np.arange(0., 1.+h, h)
  Y = np.arange(0., 1.+h, h)
  X, Y = np.meshgrid(X, Y)
  Z = solgrad.reshape(int(1./h+1),int(1./h+1))

  # Plot the surface.
  surf = ax.pcolor(X, Y, Z[:-1,:-1], shading='flat', cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

  # Create a Rectangle patch
  #rect = patches.Rectangle((0.1, 0.1), 0.2, 0.2, linewidth=0.5, edgecolor='k', facecolor='none')

  # Add the patch to the Axes
  #ax.add_patch(rect)

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  hx = 0.1
  # Add a grid
  plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)
  plt.minorticks_on()
  plt.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.2)
  major_ticks = np.arange(0., 1.+hx, hx)
  minor_ticks = np.arange(0., 1.+hx, hx/5)
  plt.xlabel('x')
  plt.ylabel('y')

  plt.xticks(major_ticks)
  plt.yticks(major_ticks)
  plt.show()


def plot_residual(res):
    fig, ax = plt.subplots()
    ax.plot(res)
    ax.set_title('Residuals of PCG')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration i')
    ax.set_ylabel('Residual')
    plt.show()
    print(f'End residual: {res[-1]}')


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patches as patches

def plot_dualsolutions(solcg, solpcg, h):
    hx = 0.1
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Increase the figure size to make the plot area wider
    ax1 = axs[0]
    ax2 = axs[1]

    fig.subplots_adjust(wspace=0.4)  # Increase the padding between subplots

    # Make data.
    X = np.arange(0., 1.+h, h)
    Y = np.arange(0., 1.+h, h)
    X, Y = np.meshgrid(X, Y)
    Z1 = solcg.reshape(int(1./h+1),int(1./h+1))
    Z2 = solpcg.reshape(int(1./h+1),int(1./h+1))

    # Plot the surface.
    surf1 = ax1.pcolor(X, Y, Z1[:-1,:-1], shading='flat', cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    surf2 = ax2.pcolor(X, Y, Z2[:-1,:-1], shading='flat', cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Add a grid to ax1
    ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)
    ax1.minorticks_on()
    ax1.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.2)
    ax1.set_xticks(np.arange(0., 1.+hx, hx))
    ax1.set_yticks(np.arange(0., 1.+hx, hx))

    # Add a color bar which maps values to colors.
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    hx = 0.1
    # Add a grid
    ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)
    ax1.minorticks_on()
    ax1.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.2)
    ax1.set_xticks(np.arange(0., 1.+hx, hx))
    ax1.set_yticks(np.arange(0., 1.+hx, hx))
    ax1.set_title('CG Solution')

    ax2.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)
    ax2.minorticks_on()
    ax2.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.2)
    ax2.set_xticks(np.arange(0., 1.+hx, hx))
    ax2.set_yticks(np.arange(0., 1.+hx, hx))
    ax2.set_title('PCG Solution')

    plt.show()


def compare_residuals(res, res0, res1, h):
    hx = 0.1
    fig, ax = plt.subplots()
    ax.plot(res)
    ax.plot(res0)
    ax.plot(res1)
    ax.legend(['mux = 1', 'mux = 0.1', 'mux = 0.01'])
    ax.set_title('Residuals of PCG')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration i')
    ax.set_ylabel('Residual')
    plt.show()

def compare_residuals(res, res0, res1, h):
    hx = 0.1
    fig, ax = plt.subplots()
    ax.plot(res)
    ax.plot(res0)
    ax.plot(res1)
    ax.legend(['PCG $\mu_x = 1$', 'PCG $\mu_x = 0.1$', 'PCG $\mu_x = 0.01$'])
    ax.set_title('Residuals of PCG')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration i')
    ax.set_ylabel('Residual')
    plt.show()

def compare_residuals2(rescg, rescg0, rescg1, res, res0, res1, h):
    hx = 0.1
    fig, ax = plt.subplots()
    ax.plot(rescg, linestyle='--')
    ax.plot(rescg0, linestyle='--')
    ax.plot(rescg1, linestyle='--')
    ax.plot(res)
    ax.plot(res0)
    ax.plot(res1)
    ax.legend(['CG $\mu_x = 1$','CG $\mu_x = 0.1$','CG $\mu_x = 0.01$','PCG $\mu_x = 1$', 'PCG $\mu_x = 0.1$', 'PCG $\mu_x = 0.01$'])
    ax.set_title('Residuals of PCG')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration i')
    ax.set_ylabel('Residual')
    plt.show()
