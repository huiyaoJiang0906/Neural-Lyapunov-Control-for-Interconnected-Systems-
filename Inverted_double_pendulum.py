import torch
import torch.nn.functional as F
import numpy as np
import timeit
from torch.nn.functional import tanh, relu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF
import math
import torch.linalg as linalg   
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def f_value(x1, x2, u):
    G = 9.8
    L1 = 0.5
    L2 = 0.5
    m1 = 1
    m2 = 1
    b1 = 0.009
    b2 = 0.009
    a1 = 0.1
    a2 = 0.1
    k = 30
    d = 0.1
    l0 = 1
    y1 = torch.zeros_like(x1)
    y2 = torch.zeros_like(x2)
    y1[:,0] = x1[:,1]
    y1[:,1] = (m1 * G * L1 * np.sin(x1[:,0]) - b1 * x1[:,1]
              + k * (1 + d**2 * (((l0 + a2*np.sin(x1[:,0]) - a1 * np.sin(x1[:,0]))**2 + (a1*np.cos(x1[:,0])
              - a2 * np.cos(x2[:,0]))**2)**0.5-l0)**2) * (((l0 + a2 * np.sin(x1[:,0]) - a1 * np.sin((x1[:,0])))**2 + (a1*np.cos(x1[:,0]) - a2 * np.cos(x2[:,0]))**2)**0.5 - l0) * a1 * np.cos(x1[:,0] - np.arctan((a1 * np.cos(x1[:,0])-a2 * np.cos(x2[:,0])) / (l0 - a1 * np.sin(x1[:,0])+a2 * np.sin(x2[:,0]))))) / (m1 * L1 ** 2)

    y2[:,0] = x2[:,1]
    y2[:,1] = (m2 * G * L2 * np.sin(x2[:,0]) - b2 * x2[:,1] + k * (1 + d**2 * (((l0 + a2*np.sin(x1[:,0]) - a1 * np.sin((x1[:,0])))**2 + (a1*np.cos(x1[:,0])
               - a2 * np.cos(x2[:,0]))**2)**0.5-l0)**2) * (((l0 + a2 * np.sin(x1[:,0]) - a1 * np.sin((x1[:,0])))**2 + (a1 * np.cos(x1[:,0]) - a2 * np.cos(x2[:,0]))**2)**0.5 - l0) * a2 * np.cos(x2[:,0] - np.arctan((a1 * np.cos(x1[:,0])-a2 * np.cos(x2[:,0])) / (l0 - a1 * np.sin(x1[:,0])+a2 * np.sin(x2[:,0]))))) / (m2 * L2 ** 2)

    y1[:, 1] = y1[:, 1] + (u[:, 0] / (m1 * L1 ** 2))
    y2[:, 1] = y2[:, 1] + (u[:, 0] / (m2 * L2 ** 2))
    return y1, y2

class fNet(torch.nn.Module):
    def __init__(self,n_input, n_hidden1,  n_output):
        super().__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden1)
        self.layer2 = torch.nn.Linear(n_hidden1,n_output)
        self.to(device)

    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        return out

with torch.no_grad():
        torch.cuda.empty_cache()

# generate training dataset
N_f = 1801
xx = np.linspace(-0.5, 0.5, N_f, dtype=float)
x_f1 = []
x_f2 = []
for i in range(0, N_f):
    for j in range(0, N_f):
        x_f1.append([xx[j], xx[i]])

x_f1 = torch.tensor(x_f1)
x_f1 = x_f1.float()

for i in range(0, N_f):
    for j in range(0, N_f):
        x_f2.append([xx[j], xx[i]])

x_f2 = torch.tensor(x_f2)
x_f2 = x_f2.float()

ut_bdd = 5.625  # bound for input
u_t = torch.Tensor(len(x_f1), 1).uniform_(-ut_bdd, ut_bdd)
vt_bdd = 3
v_t1 = torch.Tensor(len(x_f1), 1).uniform_(-vt_bdd, vt_bdd)
# target
t_f = f_value(x_f1, x_f2, u_t)
# input of FNN
x_train = torch.cat((x_f1, x_f2, u_t), 1)
# define parameters
max_iter = 1000
losses = []

fnet = fNet(n_input=5, n_hidden1=400, n_output=2)
optimizer = torch.optim.Adam(fnet.parameters(), lr=0.06)

loss_func = torch.nn.MSELoss(reduction='sum')

# # training
for epoch in tqdm(range(max_iter)):
    x_train = x_train.to(device)
    t_f1, t_f2 = t_f[0].to(device), t_f[1].to(device)  
    y_nn = fnet(x_train)
    y_nn_device = y_nn.to(device)  

    loss = loss_func(y_nn_device[:, :1], t_f1) + loss_func(y_nn_device[:, 1:], t_f2)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        torch.cuda.empty_cache()


plt.plot(losses)
losses[-1]
# train more epoches
# fnet = fnet.to(device) switch between devices if needed
losses = []
optimizer = torch.optim.Adam(fnet.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss(reduction='sum')

for epoch in tqdm(range(1000)):
    x_train = x_train.to(device)
    t_f1, t_f2 = t_f[0].to(device), t_f[1].to(device)  
    y_nn = fnet(x_train)
    y_nn_device = y_nn.to(device) 

    loss = loss_func(y_nn_device[:, :1], t_f1) + loss_func(y_nn_device[:, 1:], t_f2)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        torch.cuda.empty_cache()
# parameters
# # infnity norm if needed
# dist = y_l-t_l
# loss_all = torch.linalg.norm(dist, float('inf'))
# torch.max(loss_all)
# save the weights to calculate the Lipschitz constant with LipSDP
r = 0.8 # region of interest
N_l = 1601 # change according to delta
xl = np.linspace(-r,r,N_l, dtype = float)
u_bdd = 8.  # bound for input
# check the rough value of alpha
x_l = []
N_l = 1601 
for i in range(0,N_l): 
    for j in range(0,N_l):
        x_l.append([xl[j],xl[i]])

x_l = torch.tensor(x_l)
x_l = x_l.float()

u_l = torch.Tensor(len(x_l), 1).uniform_(-u_bdd, u_bdd) 

# target
t_l = f_value(x_l,u_l)
x_bdd = torch.cat((x_l,u_l),1)

# output of FNN
x_bdd = x_bdd.to(device)
t_l = t_l.to(device)
# fnet = fnet.to(device) # if need to switch device
y_l = fnet(x_bdd)

# maximum of loss
loss_all = torch.norm(y_l-t_l, dim = 1)
alpha = torch.max(loss_all)
N_u = 50001  # 100001 corresponds to 1e-4
u_l = np.linspace(-u_bdd,u_bdd, N_u, dtype = float).reshape(N_u,1)
u_l = torch.tensor(u_l)
alpha = torch.zeros(N_l,N_l)

for i in tqdm(range(0,N_l)): 
    for j in range(0,N_l):
        x1 = xl[j]*torch.ones_like(u_l)
        x2 = xl[i]*torch.ones_like(u_l)
        x_l = torch.cat((x1,x2), dim = 1)
        t_l = f_value(x_l,u_l)
        X_l = torch.cat((x_l,u_l),1)
        X_l = X_l.float()
        X_l = X_l.to(device)
        t_l = t_l.to(device)
        with torch.no_grad():
            y_l = fnet(X_l)
            
        torch.cuda.empty_cache()
        # target
        loss_all = torch.norm(y_l-t_l, dim = 1)
        del y_l
        max = torch.max(loss_all)
        alpha[i,j] = max

alpha_max = torch.max(alpha)
f_w1 = fnet.layer1.weight.data.cpu().numpy()
f_w2 = fnet.layer2.weight.data.cpu().numpy()

np.savetxt("fw1.txt", f_w1, fmt="%s")
np.savetxt("fw2.txt", f_w2, fmt="%s")
# save the fNN
# torch.save(fnet.cpu(), 'PF_fnet.pt')

# load fNN from a file
# fnet = torch.load('PF_fnet.pt').to(device)

def f_learned(x1,x2,u):
    X = torch.cat((x1, x2, u),1)
    y1,y2 = fnet(X)
    return y1, y2

class L_Net(torch.nn.Module):

    def __init__(self):
        super(L_Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = nn.Linear(2, 6)
        self.layer2 = nn.Linear(6, 1)
        self.control = nn.Linear(2, 1, bias=False)
        self.layer3 = nn.Linear(4, 12)  
        self.layer4 = nn.Linear(12, 1)
    def forward(self, x_1, x_2):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x_1))
        out = sigmoid(self.layer2(h_1))
        u = self.control(x_1)
        h_2 = sigmoid(self.layer3(x_2))
        out2 = sigmoid(self.layer4(h_2))
        return out, u, out2

def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon1,epsilon2,epsilon3):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    ball= Expression(0)
    lie_derivative_of_V = Expression(0)
    #lie_derivative_of_V
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])  
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
    condition = logical_and(logical_imply(ball_in_bound,  epsilon3 >= V >= epsilon1),
                           logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon2))
    return CheckSatisfiability(logical_not(condition),config)


def AddCounterexamples(x,CE,N): 
    # Adding CE back to sample set
    c = []
    nearby= []
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.random.uniform(lb,ub,N)
        nearby.append(nearby_)
    for i in range(N):
        n_pt = []
        for j in range(x.shape[1]):
            n_pt.append(nearby[j][i])             
        x = torch.cat((x, torch.tensor([n_pt], dtype=torch.float)), 0)
    return x

def dtanh(s):
    # Derivative of activation
    return 1.0 - s**2

def Tune(x):
    # Circle function values
    y = []
    for r in range(0,len(x)):
        v = 0 
        for j in range(x.shape[1]):
            v += x[r][j]**2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y

N = 500  # sample size
c_1 = 1
c_2 = 1
c_3 = 1
torch.manual_seed(10)
x1 = torch.Tensor(N, 2).uniform_(-3, 3)
x2 = torch.Tensor(N, 2).uniform_(-3, 3)
x3 = torch.Tensor(500, 4).uniform_(-3, 3)
x_0 = torch.zeros([1, 2])
x_02 = torch.zeros([1, 4])
x_0 = x_0.to(device)
x_02 = x_02.to(device)

X1 = Variable("X1")
X2 = Variable("X2")
X3 = Variable("X3")
X4 = Variable("X4")
vars_ = [X1,X2]
G = 9.8
L1 = 0.5
L2 = 0.5
m1 = 1
m2 = 1
b1 = 0.009
b2 = 0.009
a1 = 0.1
a2 = 0.1
k = 30
d = 0.1
l0 = 1
config = Config()
config.use_polytope_in_forall = True
config.use_local_optimization = True
config.precision = 1e-2
epsilon1 = torch.mean( c_1 *torch.norm(x1, dim=1))
epsilon2 = torch.mean( c_2 *torch.norm(x1, dim=1))
epsilon3 = torch.mean( c_3 *torch.norm(x1, dim=1))
# Checking candidate V within a ball around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub)
ball_lb = 0.5
ball_ub = 3
Kf = 13
KF = 578
out_iters = 0
valid = False
while out_iters < 1 and not valid: 
    start = timeit.default_timer()
    model = L_Net()
    L = []
    i = 0 
    t = 0
    max_iters = 2000
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    f_w1 = fnet.layer1.weight.data.cpu().numpy()
    f_w2 = fnet.layer2.weight.data.cpu().numpy()
    f_b1 = fnet.layer1.bias.data.cpu().numpy()
    f_b2 = fnet.layer2.bias.data.cpu().numpy()

    while i < max_iters and not valid:
        x1 = x1.float()
        x1 = x1.to(device)
        x2 = x2.float()
        x2 = x2.to(device)        
        x3 = x3.float()
        x3 = x3.to(device)
        v_candidate, u, Ro = model(x1, x3)
        X0,u0,X0_1 = model(x_0, x_02)
        f1, f2 = f_learned(x1, x2, u)
        x1_norm = torch.norm(x1, p=2, dim=1, keepdim=True)
        Circle_Tuning = Tune(x1)
        Circle_Tuning = Circle_Tuning.to(device)
        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
        L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(v_candidate),model.layer2.weight)\
                            *dtanh(torch.tanh(torch.mm(x1,model.layer1.weight.t())+model.layer1.bias)),model.layer1.weight),f1.t()),0)

        # With tuning term 
        loss = (relu(-v_candidate + c_1 * x1_norm) + relu(v_candidate - c_2 * x1_norm) + relu(
           L_V + c_3 * x1_norm - Ro) + 1.5*relu(Ro)).mean()+1.2*((Circle_Tuning-6*v_candidate).pow(2)).mean()
        
        
        print(i, "loss=",loss.item()) 
        L.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        w1 = model.layer1.weight.data.cpu().numpy()
        w2 = model.layer2.weight.data.cpu().numpy()
        B1 = model.layer1.bias.data.cpu().numpy()
        B2 = model.layer2.bias.data.cpu().numpy()
        q = model.control.weight.data.cpu().numpy()

        # Falsification
        if i % 10 == 0:
            u_NN = (q.item(0)*X1 + q.item(1)*X2) 
            print(u_NN)
            vars_nn = [X1, X2, q.item(0)*X1 + q.item(1)*X2]
            f_h1 = []
            f_z1 = np.dot(vars_nn,f_w1.T)+f_b1
            for n in range(len(f_z1)):
                f_h1.append(tanh(f_z1[n]))
            f_learn = np.dot(f_h1, f_w2.T) + f_b2

            z1 = np.dot(vars_,w1.T)+B1

            v1 = []
            for j in range(0,len(z1)):
                v1.append(tanh(z1[j]))
            z2 = np.dot(v1,w2.T)+B2
            V_learn = tanh(z2.item(0))

            print('===========Verifying==========')        
            start_ = timeit.default_timer() 
            result= CheckLyapunov(vars_, f1, V_learn, ball_lb, ball_ub, config, epsilon1, epsilon2, epsilon3)
            stop_ = timeit.default_timer() 

            if (result): 
                print("Not a Lyapunov function. Found counterexample: ")
                print(result)
                x1 = x1.to('cpu')
                x2 = x2.to('cpu')
                x3 = x3.to('cpu')
                x1 = AddCounterexamples(x1,result,10)
                x2 = AddCounterexamples(x2,result,10)
                x3 = torch.cat((x1,x2), dim = 1)
            else: 
                M = 0.5 # lower bound of M
                violation = CheckdVdx(vars_, V_learn, ball_ub, config, M) 
                while violation:
                    violation = CheckdVdx(vars_, V_learn, ball_ub, config, M)
                    if not violation:
                        dvdx_bound = np.sqrt(M)
                        print(dvdx_bound, "is the norm of dVdx")
                    M += 0.01
                beta = -dvdx_bound*((Kf+KF)*d+loss) # update beta 
                result= CheckLyapunov(vars_, f1, V_learn, ball_lb, ball_ub, config, epsilon1, epsilon2, epsilon3) # SMT solver
                if not result_strict:
                   valid = True
                   print("Satisfy conditions!!")
                   print(V_learn, " is a Lyapunov function.")
            t += (stop_ - start_)
            print('==============================') 
        i += 1

    stop = timeit.default_timer()
    np.savetxt("w1.txt", model.layer1.weight.data.cpu(), fmt="%s")
    np.savetxt("w2.txt", model.layer2.weight.data.cpu(), fmt="%s")
    np.savetxt("B1.txt", model.layer1.bias.data.cpu(), fmt="%s")
    np.savetxt("B2.txt", model.layer2.bias.data.cpu(), fmt="%s")
    np.savetxt("q.txt", model.control.weight.data.cpu(), fmt="%s")

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)
    print(V_learn)
    
    out_iters+=1
