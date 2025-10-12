from mpl_toolkits.mplot3d import Axes3D
import scipy.special as special
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import math

R0 = 1
V0 = 50

alpha = 10
size = 16 # basis cardinality

# The potential difference
def DV0(r):
    return 0 if r < 1 else V0 * (1 - r**2)

def IWantAPlot(m, eignum, justeig=False):
    """
    Computes the wavefunction corresponding to the
    energy eigenvalue eignum and angular constant m.

    Params:
    m (int) - nonnegative int (see text)
    eignum (int) - energy eigenvalue (nonegative int)
    justeig (bool) [optional] - output just energy eigenvalues

    """
    Hm = np.zeros((size, size))

    # Normalized solutions to the generic harmonic oscillator
    def norm_lag(n, m, x):
        # Normalization constant
        Ni = np.sqrt(2 * alpha * math.factorial(n) / math.factorial(n + m))
        return Ni * np.power(x, m) * np.exp((-1)*alpha*x**2 / 2) * special.genlaguerre(n, m)(alpha*x**2)

    # Constructing Hamiltonian
    for i in range(0, size):
      for j in range(0, size):
          Hm[i][j] = integrate.quad(
              lambda x: x * norm_lag(i, m, x) * DV0(x) * norm_lag(j, m, x),
              R0, 16 * R0)[0]
          if i == j:
              Hm[i][j] += alpha * (2 * i + m + 1)

    # Eigenenergies and eigenvectors for the Hamiltonian
    eigvals, eigvecs = np.linalg.eigh(Hm)

    if justeig:
        print(eigvals)
        return 0
    
    def solution(eigvector, x):
        """Compute wavefunction solution for a given radial position x."""
        ans = np.zeros_like(x, dtype=float)
        for i in range(0, size):
            Ni = np.sqrt(2 * alpha * math.factorial(i) / math.factorial(i + m))
            ans += eigvector[i] * Ni * np.power(x, m) * np.exp((-1)*alpha*x**2 / 2) * special.genlaguerre(i, m)(alpha*x**2)
        return ans

    vfunc = np.vectorize(solution, excluded=['eigvector'])

    R = np.linspace(0, 1.5*R0, 100)
    P = np.linspace(0, 2 * np.pi, 100)
    R, P = np.meshgrid(R, P)

    Z = vfunc(eigvector=eigvecs[:, eignum], x=R) * np.exp(1j * m * P)
    Z_real = np.real(Z)

    X, Y = R * np.cos(P), R * np.sin(P)

    return X, Y, Z_real, eigvals[eignum]

layout = [
    [(0, 0)],
    [(1, 0)],
    [(0, 1), (2, 0)],
    [(1, 1), (3, 0)],
    [(0, 2), (2, 1), (4, 0)]]

fig = plt.figure(figsize=(6 * 2 * 3, 4 * len(layout)))

total_cols = max(len(r) for r in layout) * 2
total_rows = len(layout)

for row_idx, row in enumerate(layout):
    # For centering subplots
    col_offset = (total_cols - 2 * len(row)) // 2

    for col_idx, (i, j) in enumerate(row):
        # Compute data
        X, Y, Z, eigvalue = IWantAPlot(i, j)

        ax3d = fig.add_subplot(total_rows, total_cols,
                       row_idx * total_cols + col_offset + 2 * col_idx + 1,
                       projection='3d')
        ax3d.set_aspect('auto', adjustable='box')
        ax3d.set_box_aspect(None, zoom=1.5)
        ax3d.plot_surface(X, Y, Z, cmap='inferno')
        ax3d.set_title(f"n={j}, m={i}, E={round(eigvalue,3)}")
        ax3d.set_axis_off()

        ax2d = fig.add_subplot(total_rows, total_cols,
                             row_idx * total_cols + col_offset + 2 * col_idx + 2)
        ax2d.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.3)
        ax2d.set_aspect('auto', adjustable='box')
        ax2d.plot(X[0], Z[0], 'red')
        ax2d.vlines(x=1, ymin=min(Z[0]), ymax=max(Z[0]), colors=['tab:orange'], ls='--', lw=2, alpha=1)
        ax2d.set_title(f"Radial component (n={j}, m={i})")

plt.tight_layout()
# plt.savefig('figure_custom_layout.png', dpi=300)
plt.show()
