import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def create_diffusion_matrix(N, mu_x, mu_y):
    """
    Create the diffusion matrix for anisotropic diffusion with coefficients mu_x and mu_y.
    N is the number of discretization points in each dimension (excluding boundaries).
    """
    h = 1 / (N + 1)  # Grid spacing

    # Create the 1D Laplacian matrices for x and y directions
    main_diag_x = -2 * mu_x / h**2 * np.ones(N)
    off_diag_x = mu_x / h**2 * np.ones(N - 1)
    L_x = sp.diags([main_diag_x, off_diag_x, off_diag_x], [0, -1, 1], shape=(N, N))

    main_diag_y = -2 * mu_y / h**2 * np.ones(N)
    off_diag_y = mu_y / h**2 * np.ones(N - 1)
    L_y = sp.diags([main_diag_y, off_diag_y, off_diag_y], [0, -1, 1], shape=(N, N))

    # Use Kronecker products to construct the 2D diffusion matrix
    A = sp.kron(sp.eye(N), L_x) + sp.kron(L_y, sp.eye(N))
    return A

def create_source_term(N, f_value):
    """
    Create the source term vector for the grid.
    f_value is the value of the source term in the defined region.
    """
    h = 1 / (N + 1)  # Grid spacing
    b = np.zeros((N, N))

    # Define the source region (indices)
    region_start, region_end = int(0.1 / h), int(0.3 / h)
    b[region_start:region_end, region_start:region_end] = f_value

    # Flatten the source term to match the matrix vector multiplication
    return b.flatten()

def incomplete_lu(A, drop_tol):
    """
    Construct the incomplete LU decomposition with drop tolerance.
    """
    return spla.spilu(A, drop_tol=drop_tol)

def conjugate_gradient(A, b, M_inv, tol, max_iter, x0=None):
    """
    Implement the preconditioned conjugate gradient algorithm.
    A is the system matrix, b is the right-hand side vector, and M_inv is the inverse of the preconditioner.
    x0 is the initial guess, tol is the tolerance for convergence, and max_iter is the maximum number of iterations.
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    x = x0
    r = b - A.dot(x)
    z = M_inv.dot(r)
    p = z
    rs_old = np.dot(r, z)
    res = []
    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        z = M_inv.dot(r)
        rs_new = np.dot(r, z)
        res.append(np.sqrt(rs_new))
        if np.sqrt(rs_new) < tol:
            break
        
        p = z + (rs_new / rs_old) * p
        rs_old = rs_new
    print("Number of iterations:", i)
    print("Final residual:", np.sqrt(rs_new))
    fig, ax = plt.subplots()
    ax.plot(res)
    ax.set_title('Residuals of PCG')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration i')
    ax.set_ylabel('Residual')
    plt.show()

    return x

# Parameters
h = 0.01
N = int(1./h + 1)
n = N**2  # Number of discretization points in each dimension
mu_x, mu_y = 1.0, 1.0  # Diffusion coefficients
tol = 1e-6  # Tolerance for convergence
max_iter = 1000  # Maximum number of iterations
f_value = 1.0  # Source term value
drop_tol = 0.1  # Drop tolerance for ILU

# Create the matrix and source term
A = -create_diffusion_matrix(N, mu_x, mu_y)
b = create_source_term(N, f_value)

# Construct the ILU preconditioner
M_ilu = incomplete_lu(A, drop_tol)
M_inv = spla.LinearOperator(A.shape, M_ilu.solve)

# Solve the system using the PCG algorithm
solution = conjugate_gradient(A, b, M_inv, tol=tol, max_iter=max_iter)

# Reshape the solution for visualization
solution_2d = solution.reshape((N, N))
solution_2d.shape, solution_2d.min(), solution_2d.max()

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import matplotlib.patches as patches

fig, ax = plt.subplots()

# Make data.
X = np.arange(0., 1.+h, h)
Y = np.arange(0., 1.+h, h)
X, Y = np.meshgrid(X, Y)
Z = solution_2d

# Plot the surface.
surf = ax.pcolor(X, Y, Z[:-1,:-1], shading='flat', cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Create a Rectangle patch
rect = patches.Rectangle((0.1, 0.1), 0.2, 0.2, linewidth=1, edgecolor='k', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
hx = 0.1
# Add a grid
plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)
plt.minorticks_on()
plt.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.2)
major_ticks = np.arange(0., 1.+hx, hx)
minor_ticks = np.arange(0., 1.+hx, hx/5)

plt.xticks(major_ticks)
plt.yticks(major_ticks)
plt.show()

L = M_ilu.L
U = M_ilu.U

fig, axs = plt.subplots(1, 3)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

ax1.spy(A, markersize=1)
ax1.set_title('A (nnz = ' + str(A.getnnz()) + ')')
ax2.spy(L, markersize=1)
ax2.set_title('L (nnz = ' + str(L.getnnz()) + ')')
ax3.spy(U, markersize=1)
ax3.set_title('U (nnz = ' + str(U.getnnz()) + ')')

plt.show()
