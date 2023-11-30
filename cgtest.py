import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

# Create a sparse matrix
A = csr_matrix([[3, 1, 0], [1, 2, 1], [0, 1, 4]])

# Create a right-hand side vector
b = np.array([1, 2, 3])

# Solve the linear system using cg
x, info = cg(A, b)

# Print the solution
print("Solution:", x)
