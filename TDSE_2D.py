import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import time
import numba

# Exact solution
@numba.njit
def Phi_exact(x, t, x0, sigma, p0, hbar, m):
    exp = np.exp( (- (1j*t*p0**2)/(2*m*hbar) + (1j*p0*(x-x0))/hbar - (x-x0)**2/(4*sigma**2)) / (1 + ((1j*t*hbar)/(2*m*sigma**2))) )
    return exp / ( ((2*np.pi)**0.25) * ((sigma**2)**0.25) * np.sqrt(1 + (1j*t*hbar)/(2*m*sigma**2)) )

# Wave packet at t = 0 
def Capital_Phi(x, x0, sigma, p0, hbar, m):
    return Phi_exact(x, 0, x0, sigma, p0, hbar, m)

# Initial wave function in momentum space
@numba.njit
def A(x, x0, sigma, p0, hbar, m):
    exp = np.exp( ((1j*p0*(x-x0))/hbar - (x-x0)**2/(4*sigma**2)) )
    return exp / ( ((2*np.pi)**0.25) * ((sigma**2)**0.25))

# Energy carried by the wave packet
@numba.njit
def Energy(sigma, p0, hbar, m):
    return p0**2/(2*m) + hbar**2/(8*m*sigma**2)

def Potential(x, V0, delta):
    return np.piecewise(x, [np.abs(x) <= delta/2], [V0, 0])

'''
# Reflectionless potential
def Potential(x, V0, delta):
    return -h_squared/2/m/alpha**2*lamb*(lamb + 1)/ np.cosh(x/alpha)**2
'''

# Normalize 
def normalize(func):
    
    norm = np.linalg.norm(func)
    return func / norm if norm != 0 else func


start = time.time()

# Numerical parameters
hbar = 1; m = 1;
x0 = -30; p0 = 7.5; V0 = - 28.2;     
sigma = 2; delta = 2;
nmax = 3000; dt = 1/50; h = 1/25;
lamb = 20; alpha = 2;

# Numerical integration
xx = np.arange(-nmax * h, (nmax + 1) * h, h)
h_squared = h**2
factor = 8j * h_squared / dt

U = np.array([Potential(i * h, V0, delta) for i in range(-nmax, nmax + 1)])
 
# Matrix A
matA =  diags([1, -2 + 4j * h**2 / dt - 2 * h**2 * U, 1], [-1, 0, 1], shape=(2 * nmax + 1, 2 * nmax + 1), format='csc')
# Sparse matrix A
sparse_matA = csr_matrix(matA)

# Initialize sol_phi
sol_phi = np.array([Capital_Phi(j * h, x0, sigma, p0, hbar, m) for j in range(-nmax, nmax + 1)])

# Create the figure and axes 
fig, ax = plt.subplots(figsize=(12, 8))
line, = ax.plot(xx, np.abs(sol_phi)**2, label="Numerical")
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$|\\Psi(x)|^2$")

barrier_rect = Rectangle((-delta/2, 0), delta, V0, alpha=0.2, color='gray')
ax.add_patch(barrier_rect)

ax.set_xlim(-nmax*h, nmax*h)
ax.set_ylim(0, 0.2)
time_text = ax.text(0.15, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12)

# Animation function
def update(frame):
    global sol_phi
    
    
    matB = factor * sol_phi
    Q = spsolve(sparse_matA, matB) - sol_phi
    sol_phi = Q.copy()

    # Normalize 
    sol_phi_normalized = normalize(np.abs(sol_phi)**2)
    
    line.set_ydata(sol_phi_normalized)
    time_text.set_text(f'Time: {frame * dt:.2f}')
    
    return line, time_text

# Create animation
num_frames = 1000
animation = FuncAnimation(fig, update, frames=num_frames, interval=20, blit=True, repeat=False)

end = time.time()
print('time =', end-start)
plt.show()
