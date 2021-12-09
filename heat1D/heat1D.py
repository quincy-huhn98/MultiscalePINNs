import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from models_tf import Sampler, heat1D_NN, heat1D_FF, heat1D_ST_FF
import time

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

class HeatCond:
    def __init__(self, cond, qext, dx, udir, n_ref):
        self.cond = cond
        self.qext = qext
        self.dx = dx
        self.udir = udir
        
        self.repeat_properties(n_ref)
        self.init_mesh()
        self.init_mat_ind()
        
        self.build_LHS_matrix()
        self.build_rhs()
  
    def repeat_properties(self, n_ref):
        # Replecate per zone properties to assign to a mesh
        self.cond = np.repeat(self.cond, n_ref) 
        self.qext = np.repeat(self.qext, n_ref)
        self.dx   = np.repeat(self.dx  , n_ref) / float(n_ref)

    def init_mesh(self):
        # cell interfaces
        x = np.zeros(len(self.dx)+1)
        for i in range(len(self.dx)):
            x[i+1] = x[i] + self.dx[i]
        self.x = x
        # cell mid-point
        xm = x[:-1] + self.dx/2.
        # number of cells, pts
        self.n_cells = len(self.dx)
        self.n_pts   = self.n_cells + 1
  
    def init_mat_ind(self):
        # create indices for rows and columns of the tridiagonal stifness matrix
        indr = np.kron(np.arange(0,self.n_pts),np.ones(3))
        self.indr = indr[1:-1]
        
        indc = np.zeros((self.n_pts,3))
        indc[0,:] = np.array([-1,0,1],dtype=int)
        for i in range (1,self.n_pts):
            indc[i,:] = indc[i-1,:]+1
        indc = indc.flatten()
        self.indc = indc[1:-1]

    def build_LHS_matrix(self):
        # matrix
        L = np.zeros(self.n_pts)
        L[1:] = -self.cond/self.dx
        R = np.zeros(self.n_pts)
        R[:-1] = -self.cond/self.dx
        D = np.zeros(self.n_pts)
        D[:-1] -= R[:-1] 
        D[1:]  -= L[1:] 
        arr = np.vstack((L,D,R)).T.flatten()[1:-1]
        self.A = csr_matrix( (arr, (self.indr, self.indc)), shape=(self.n_pts, self.n_pts) )

    
    def apply_dirchilet(self):
        # apply right bc (Dirichlet)
        self.A[-1,-2:]=0
        self.A[-1,-1]=1
        self.A[0,1]=0
        self.A[0,0]=1

    def build_rhs(self):
        # forward rhs
        self.rhs = np.zeros(self.n_pts)
        self.rhs[:-1] += self.qext*self.dx/2
        self.rhs[1:]  += self.qext*self.dx/2
        # apply bc
        self.rhs[-1]= self.udir
        self.rhs[0]= self.udir

    def solve(self):
        # solve
        self.u  = spsolve(self.A, self.rhs)
        return self.u, self.x

    def evaluate_u(self, point):
        f = interp1d(self.x, self.u)
        return f(point)


if __name__ == '__main__':
    start_time = time.time()
    # Define exact solution
    def u(x, a, b):
        """
        :param x: x = (t, x)
        """
        Diff = np.array([a,a])
        qext = np.array([b[0],b[1]]) 
        dx   = np.array([0.5,0.5])

        Cond = HeatCond(Diff,qext,dx,0,100)
        Cond.apply_dirchilet()
        _, _ = Cond.solve()

        return Cond.evaluate_u(x)

    def f(x, a, b):
        mask = np.zeros(np.shape(x))
        for i in range(len(x)):
            if x[i] < 0.5:
                mask[i] = [1]
        mask *= 250
        mask += 250
        return mask
    
    # Define PDE residual
    #def operator(u, t, x, k,  sigma_t=1.0, sigma_x=1.0):
    def operator(u, x, k, sigma_x=1.0):
        #u_t = tf.gradients(u, t)[0] / sigma_t
        u_x = tf.gradients(u, x)[0] / sigma_x
        u_xx = tf.gradients(u_x, x)[0] / sigma_x
        #residual = u_t - k * u_xx
        residual = - k * u_xx
        return residual

    # Parameters of equations
    a = 5
    b = [500,250]
    k = a
    q = b

    # Domain boundaries
    bc1_coords = np.array([0.0])
    bc2_coords = np.array([1.0])
    dom_coords = np.array([0.0,1.0])

    # Create initial conditions samplers
    # ics_sampler = Sampler(2, ics_coords, lambda x: u(x, a, b), name='Initial Condition 1')

    # Create boundary conditions samplers
    bc1 = Sampler(1, bc1_coords, lambda x: u(x, a, b), name='Dirichlet BC1')
    bc2 = Sampler(1, bc2_coords, lambda x: u(x, a, b), name='Dirichlet BC2')
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(1, None, lambda x: f(x, a, b), name='Forcing')

    # Test data
    nn = 100  # nn = 1000
    #t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x = np.linspace(dom_coords[0], dom_coords[1], nn)[:, None]
    #t, x = np.meshgrid(t, x)
    X_star = np.hstack((x.flatten()[:, None]))
    X_star = X_star.reshape(-1,1)
    u_star = u(X_star, a, b)
    f_star = f(X_star, a, b)

    # Define model
    # heat1D_NN: Plain MLP
    # heat1D_FF: Plain Fourier feature network
    # heat1D_ST_FF: Spatial-temporal Plain Fourier feature network
    
    layers = [1, 100, 100, 100, 1]  # For heat1D_NN, use layers = [2, 100, 100, 100, 1]
    sigma = 500   # Hyper-parameter for Fourier feature embeddings
    model = heat1D_NN(layers, operator, k, q,
                             bcs_sampler, res_sampler, 
                             sigma, X_star, u_star)

    # Train model
    model.train(nIter=10000, batch_size=128)
    print(time.time()-start_time)

    # Predictions
    u_pred = model.predict_u(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Relative L2 error_u: {:.2e}'.format(error_u))
    

    # Grid data
    U_star = griddata(X_star, u_star.flatten(), (x), method='cubic')
    #F_star = griddata(X_star, f_star.flatten(), (x), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (x), method='cubic')
    
    # Plot
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x, U_star)
    plt.xlabel('$x$')
    plt.ylabel('$T$')
    plt.title(r'Exact')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.plot(x, U_pred)
    plt.xlabel('$x$')
    plt.ylabel('$T$')
    plt.title(r'Predicted')
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.plot(x, np.abs(U_star - U_pred))
    plt.xlabel('$x$')
    plt.ylabel('$T$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig('heterogenous_ff.jpg')

    #loss_ics = model.loss_ics_log
    loss_bcs = model.loss_bcs_log
    loss_res = model.loss_res_log
    l2_error = model.l2_error_log
    
    fig_2 = plt.figure(2, figsize=(6, 5))
    with sns.axes_style("darkgrid"):
        iters = 100 * np.arange(len(loss_res))
            
        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$', linewidth=2)
        plt.plot(iters, loss_bcs, label='$\mathcal{L}_{bc}$', linewidth=2)
        #plt.plot(iters, loss_ics, label='$\mathcal{L}_{ic}$', linewidth=2)
        plt.plot(iters, l2_error, label=r'$L^2$ error', linewidth=2)
        
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2, fontsize=17)
        plt.tight_layout()
        plt.show()

        
    
