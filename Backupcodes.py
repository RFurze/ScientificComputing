#Check A
h = 0.02
mux=1.0
muy=1.0
drop_tol = 0.1
n = int(1./h + 1)
N = n**2

A = diffusionMatrix(mux, muy, h)
intNodes, extNodes = boundaryConditions(int(1./h+1))
A_int = A[intNodes][:, intNodes]
M = sp.sparse.linalg.spilu(A_int, drop_tol=drop_tol)
L = M.L
U = M.U
A.todense()

print(f'Number of nonzero elements in L: {L.nnz}')
print(f'Number of nonzero elements in U: {U.nnz}')
print(f'Unconditioned sparsity of A: {(1-(A.nnz)/(N*N) )*100:.2f}%')
print(f'Conditioned sparsity of ILU factored A {(1-(L.nnz+U.nnz)/(N*N) )*100:.2f}%')

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
