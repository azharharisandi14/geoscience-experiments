import numpy as np 
import matplotlib.pyplot as plt 
from copy import deepcopy
import matplotlib.animation as animation 

def InitialCondition(x, b=0.5):
    return np.exp(-(x-b)**2/(2e-3))

def absorbing_boundary(u, x, nxb, alpha=0.8):
    """nxb : jumlah grid absorbing dari ujung grid
    contoh : jika nxb = 5, dan jumlah grid n, 
    grid yg absorbing ada di grid 0, 1, 2, 3,4
    dan grid n, n-1, n-2, n-3, n-4
    """
    T = len(x[:nxb])
    diff = x[nxb]-x[:nxb]
    f_left = np.exp(-np.abs((diff/T*alpha)))
    f_right = np.flip(f_left)

    u[:nxb] = u[:nxb]*f_left
    u[-nxb:] = u[-nxb]*f_right        
    return u

def solver(IC, dx, v, dt, T, xmin, xmax):
    ts = np.arange(0, T+dt, dt) # time points
    x = np.arange(xmin, xmax+dx, dx) # spatial grid
    C = v*dt/dx # Courant number
    
    # initial condition
    u_current = IC(x)

    fig = plt.figure(figsize=(12,8))
    title = fig.suptitle('Time = 0 ms')
    ax1 = fig.add_subplot(1,1,1)
    line1, = ax1.plot(x, u_current, color='blue', lw=1.5)

    # for n, t in enumerate(ts):
    def animate(n):
        nonlocal u_current, dx, dt, T, xmin, xmax, C, x 
        # Neumann time BC
        u_past = deepcopy(u_current)

        u_future = np.zeros_like(x)
        # first time step
        if n == 0:
            # initial condition
            u_future[1:-1] = u_current[1:-1] + C**2 * (u_current[:-2] - 2*u_current[1:-1] + u_current[2:])

        # rest of the time steps
        else:
            u_future[1:-1] = -u_past[1:-1] + 2*u_current[1:-1] + C**2 * (u_current[:-2] - 2*u_current[1:-1] + u_current[2:])
        
        # apply absorbing boundary 
        u_future = absorbing_boundary(u_future, x, 7)

        # update reference
        u_past = deepcopy(u_current) 
        u_current = deepcopy(u_future)

        line1.set_ydata(u_current)
        title.set_text(f'Time = {n*dt:.2f} s')
        
        return line1

    ani = animation.FuncAnimation(fig, animate, interval=.001, frames=10000, repeat=False)
    plt.show()

if __name__ == "__main__":
    xmin, xmax = 0, 1
    tmax = 1 
    dt = 0.005 
    dx = 0.01
    v = 1

    snaps = solver(InitialCondition, dx, v, dt, tmax, xmin, xmax)


    ## Alternative plotting
    # plt.figure(figsize=(3, 9))
    # i = 1
    # n = len(snapshots)

    # for snap, label in zip(snaps, snapshots):
    #     x, amp = snap 
    #     plt.subplot(n, 1, i)
    #     plt.plot(x, amp)
    #     plt.ylim(-1.1,1.1)
    #     plt.title(f'Time : {label} s')
    #     i += 1
    
    # plt.tight_layout()
    # plt.show()
