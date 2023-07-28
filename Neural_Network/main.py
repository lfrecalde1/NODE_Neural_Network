from argparse import ArgumentParser
from collections import defaultdict
from math import ceil, sin
from math import floor
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.modules.container import T

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib as mpl
from fancy_plots import fancy_plots_1
from fancy_plots import fancy_plots_2

from casadi import MX, vertcat, sin, cos
from casadi import Function

def f_d(x, ts, f_system):
    k1 = f_system(x)
    k2 = f_system(x+(ts/2)*k1)
    k3 = f_system(x+(ts/2)*k2)
    k4 = f_system(x+(ts)*k3)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    aux_x = np.array(x[:,0]).reshape((2,))
    return aux_x

def model_system():
    g = 1.0  # gravity
    l = 1.0  # length pendulum
    b = 0.5  # System viscosity

    omega = MX.sym('omega')
    theta = MX.sym('theta')
    
    theta_d = MX.sym('theta_d')
    omega_d = MX.sym('omega_d')

    # Vector of general states of the system
    x = vertcat(theta, omega)
    xdot =vertcat(theta_d, omega_d)
    f_expl = vertcat(omega, 
                    -(g/l)*np.sin(theta) - b*omega)
    f_system = Function('system', [x], [f_expl])
    return f_system

def f(t, y):
    theta, omega = y
    g = 1.0
    l = 1.0
    b = 0.5
    omega_p = -(g/l)*np.sin(theta) - b*omega
    theta_p = omega
    return theta_p, omega_p

def xavier_init(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

def construct_network(input_dim, output_dim, hidden_dim, hidden_layers, device):
    layers = [nn.Linear(input_dim, hidden_dim), nn.Softplus()]
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Softplus()])
    layers.append(nn.Linear(hidden_dim, output_dim))

    net = nn.Sequential(*layers).double().to(device)
    xavier_init(net)
    return net

def listify(A):
        return [a for a in A.flatten()]

def plot_predictions(model, y_pred, t_eval, res, step_size, subsample_every):
    theta = res[0,:]
    omega = res[1,:]

    omega_numerical = np.diff(y_pred[:1]) / step_size
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.canvas.manager.set_window_title(model)
    ax1.set_ylabel("theta(t)")
    ax2.set_ylabel("omega(t)")
    ax2.set_xlabel("t")

    ax1.plot(t_eval, theta, c="black", label="true")
    ax1.plot(t_eval, y_pred[0], c="b", linestyle="--", label="predicted")

    ax2.plot(t_eval, omega, c="black", label="true")
    ax2.plot(t_eval, y_pred[1], c="r", linestyle="--", label="predicted")
    ax2.plot(t_eval[1:],
            omega_numerical.T,
            c="r",
            linestyle="dotted",
            label="numerical",)
    ax1.scatter(t_eval[::subsample_every],
        res[:, ::subsample_every][0],
        c="black",
        linestyle="None",
        label="collocation point",)
    ax2.scatter(
        t_eval[::subsample_every],
        res[:, ::subsample_every][1],
        c="black",
        linestyle="None",
        label="collocation point",)
    ax2.legend()
    plt.show()

def main(args):
    # Setup Experiments
    # Initial Conditions of the system
    y0 = [np.pi/4, 0]

    # Times configuration
    step_size = args.step_size
    t_start = args.t_start
    t_end = args.t_end
    t_span = (t_start, t_end + step_size)

    # Get sample time vector
    t_eval = np.arange(t_start, t_end + step_size, step_size)

    # Get model for the Neural Network
    model = args.model
    n_epochs = args.n_epochs
    device = args.device

    # Get sub sample time to get measurements
    subsample_every = int(1/step_size)

    # Initial condition runge 4
    x = np.zeros((2, t_eval.shape[0]), dtype = np.double)
    x[0, 0] = y0[0]
    x[1, 0] = y0[1]

    # System Function
    f_system = model_system()


    # Sample time
    delta_t = np.zeros((1, t_eval.shape[0]), dtype=np.double)

    for k in range(0, t_eval.shape[0]-1):

        # Get Computational Time
        tic = time.time()

        # System Evolution
        x[:, k+1] = f_d(x[:, k], step_size, f_system)
        x[:, k+1] = x[:, k+1] +  np.random.uniform(low=-0.002, high=0.002, size=(2,))

        toc = time.time()- tic
        delta_t[:, k] = toc


    # Solver using scipy
    res = solve_ivp(f, t_span = t_span, t_eval = t_eval, y0 = y0, method = "RK45")
    states = res.y

    hidden_dim = args.hidden_dim
    hidden_layers = args.n_layers
    t = torch.tensor(t_eval, device = device, requires_grad = True)

    losses = defaultdict(lambda: defaultdict(list))
    y_train = torch.tensor(x[:, ::subsample_every]).to(device)
    t_train = torch.tensor(t_eval[::subsample_every], requires_grad = True).to(device)

    ## Trainning section
    nn_vanilla = construct_network(1, 2, hidden_dim, hidden_layers, device)

    opt_vanilla = torch.optim.Adam(nn_vanilla.parameters())
    tic = time.time()
    for epoch in tqdm(range(n_epochs), desc="vanilla: training epoch"):
        out = nn_vanilla(t_train.unsqueeze(-1)).T

        loss_collocation = F.mse_loss(out, y_train)

        loss_collocation.backward()
        opt_vanilla.step()
        nn_vanilla.zero_grad()
        losses["vanilla"]["collocation"].append(loss_collocation.item())
    
    toc = time.time()- tic
    print(toc)
    y_predict = nn_vanilla(t.unsqueeze(-1)).detach().detach().cpu().T
    plot_predictions(model = model, y_pred = y_predict, t_eval = t_eval, res = x, step_size = step_size, subsample_every = subsample_every)
    

    # Figures system
    fig2, ax12, ax22 = fancy_plots_2()

    ## Axis definition necesary to fancy plots
    ax12.set_xlim((t_eval[0], t_eval[-1]))
    ax22.set_xlim((t_eval[0], t_eval[-1]))
    ax12.set_xticklabels([])

    state_1, = ax12.plot(t_eval,states[0,0:t_eval.shape[0]],
                    color='#8193F0', lw=2, ls="-")
                    
    state_1_kuta, = ax12.plot(t_eval,x[0,0:t_eval.shape[0]],
                    color='#445C72', lw=2, ls="--")

    state_2, = ax22.plot(t_eval,states[1,0:t_eval.shape[0]],
                    color='#9e4941', lw=2, ls="-")

    state_2_kuta, = ax22.plot(t_eval,x[1,0:t_eval.shape[0]],
                    color='#5D433A', lw=2, ls="--")

    ax12.set_ylabel(r"$[rad]$", rotation='vertical')
    ax12.legend([state_1, state_1_kuta],
            [r'$\theta$', r'$\theta-kuta$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax22.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax22.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax22.legend([state_2, state_2_kuta],
            [r'$\omega$', r'$\omega-kuta$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax22.grid(color='#949494', linestyle='-.', linewidth=0.5)
    fig2.savefig("system_states_noise.eps")
    fig2.savefig("system_states_noise.png")
    plt.show()
    return None

if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument(
            "--model",
            choices=["vanilla", "autodiff", "pinn"],
            default="vanilla",
            )
        parser.add_argument("--hidden_dim", default = 32, type = int, help = "Number of neurons in the hidden")
        parser.add_argument("--n_layers", default = 5, type = int, help = "Number of neural networks layer")
        parser.add_argument("--device", default = "cuda:0", help = "Selection of the CPU or GPU")
        parser.add_argument("--n_epochs", default = 2000, type = int, help = "Number of trainning cyles")
        parser.add_argument("--t_start", default = 0.0, type = float, help = "Initial time of the system")
        parser.add_argument("--t_end", type = float, default = 15.0, help = "Simulation final time")
        parser.add_argument("--step_size", type = float, default = 0.01, help = "Sample time defined for the ODE")
        args = parser.parse_args()
        main(args)
        pass

    except(KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass
