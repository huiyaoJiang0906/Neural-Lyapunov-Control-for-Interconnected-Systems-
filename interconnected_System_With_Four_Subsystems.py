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
from dreal import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def f_value(x1,x2,x3,x4,u1,u2,u3,u4,v1,v2,v3,v4):
    y1 = []
    y2 = []
    y3 = []
    y4 = []

    for r in range(0, len(x3)):
    # Define the system dynamics
       
        f1 = [-x1[r][1], 
              x1[r][0] * x1[r][1] + (2 - np.cos(x1[r][0])**2)]
        y1.append(f1)

        f2 = [x2[r][1], 
              -x2[r][0] - 0.5 * x2[r][1] + 0.5 * x2[r][0]**2 * x2[r][1] + (1 + np.cos(x2[r][1]))]
        y2.append(f2)

        f3 = [-x3[r][0] + x3[r][1], 
              -0.5 * (x3[r][0] + x3[r][1]) + (0.5 * x3[r][1] * (2 + np.cos(x3[r][0])**2)**2 + 2)]
        y3.append(f3)
        f4 = [x4[r][1], 
              -x4[r][0] - 8 * x4[r][1]]
        y4.append(f4)
 
    y1 = torch.tensor(y1)
    y2 = torch.tensor(y2)
    y3 = torch.tensor(y3)
    y4 = torch.tensor(y4)
    y1[:, 1] = y1[:, 1] + (2 - np.cos(x1[r][0])**2) * (u1[:, 0]+v1[:, 0])
    y2[:, 1] = y2[:, 1] + (1 + np.cos(x2[r][1])) * (u2[:, 0]+v2[:, 0])
    y3[:, 1] = y3[:, 1] + (0.5 * x3[r][1] * (2 + np.cos(x3[r][0])**2)**2 + 2) * (u3[:, 0]+v3[:, 0])
    y4[:, 1] = y4[:, 1] + u4[:, 0] + v4[:, 0]
    
    return y1,y2,y3,y4
#(x1[r][1] * np.sin(x2[r][0]) + x1[r][0] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][0]))
#(x2[r][0] + x2[r][1] * np.cos(x1[r][1]**2) + x2[r][1] * np.sin(x3[r][1]))
#(x3[r][0] * np.cos(x1[r][0]) + x3[r][1] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][1]))
#(x4[r][0] * np.sin(x3[r][0]) + x4[r][1] * np.cos(x2[r][0]))

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
N_f = 1500
xx = np.linspace(-6.5, 6.5, N_f, dtype=float)
x_f1 = []
x_f2 = []
x_f3 = []
x_f4 = []
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
 
for i in range(0, N_f):
    for j in range(0, N_f):
        x_f3.append([xx[j], xx[i]])

x_f3 = torch.tensor(x_f3)
x_f3 = x_f3.float()

for i in range(0, N_f):
    for j in range(0, N_f):
        x_f4.append([xx[j], xx[i]])

x_f4 = torch.tensor(x_f4)
x_f4 = x_f4.float()


ut_bdd = 5.625  # bound for input
u_t1 = torch.Tensor(len(x_f1), 1).uniform_(-ut_bdd, ut_bdd)
u_t2 = torch.Tensor(len(x_f2), 1).uniform_(-ut_bdd, ut_bdd)
u_t3 = torch.Tensor(len(x_f3), 1).uniform_(-ut_bdd, ut_bdd)
u_t4 = torch.Tensor(len(x_f4), 1).uniform_(-ut_bdd, ut_bdd)

vt_bdd = 1
v_t1 = torch.Tensor(len(x_f1), 1).uniform_(-vt_bdd, vt_bdd)
v_t2 = torch.Tensor(len(x_f2), 1).uniform_(-vt_bdd, vt_bdd)
v_t3 = torch.Tensor(len(x_f3), 1).uniform_(-vt_bdd, vt_bdd)
v_t4 = torch.Tensor(len(x_f4), 1).uniform_(-vt_bdd, vt_bdd)

# target
t_f = f_value(x_f1, x_f2, x_f3, x_f4, u_t1,u_t2,u_t3,u_t4, v_t1, v_t2, v_t3, v_t4)
# input of FNN
x_train = torch.cat((x_f1, x_f2, x_f3, x_f4, u_t1,u_t2,u_t3,u_t4, v_t1, v_t2, v_t3, v_t4), 1)
# define parameters
max_iter = 1000
losses = []
# NN: 1 hidden layers with 600 neurons
fnet = fNet(n_input=16, n_hidden1=400, n_output=8)
optimizer = torch.optim.Adam(fnet.parameters(), lr=0.06)

loss_func = torch.nn.MSELoss(reduction='sum')

# # training
for epoch in tqdm(range(max_iter)):
    x_train = x_train.to(device)
    t_f1, t_f2, t_f3, t_f4 = t_f[0].to(device), t_f[1].to(device), t_f[2].to(device), t_f[3].to(device)  
    y_nn = fnet(x_train)
    y_nn_device = y_nn.to(device)  

    loss = loss_func(y_nn_device[:, :1], t_f1) + loss_func(y_nn_device[:, :1], t_f2)+ loss_func(y_nn_device[:, :1], t_f3)+ loss_func(y_nn_device[:, :1], t_f4)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        torch.cuda.empty_cache()


losses[-1]
# train more epoches
# fnet = fnet.to(device) switch between devices if needed
losses = []
optimizer = torch.optim.Adam(fnet.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss(reduction='sum')

for epoch in tqdm(range(1000)):
    x_train = x_train.to(device)
    t_f1, t_f2,t_f3,t_f4 = t_f[0].to(device), t_f[1].to(device), t_f[2].to(device), t_f[3].to(device)  
    y_nn = fnet(x_train)
    y_nn_device = y_nn.to(device)  

    loss = loss_func(y_nn_device[:, :1], t_f1) + loss_func(y_nn_device[:, :1], t_f2)+ loss_func(y_nn_device[:, :1], t_f3)+ loss_func(y_nn_device[:, :1], t_f4)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        torch.cuda.empty_cache()
plt.plot(losses)
plt.plot(losses[-500:-1])
# parameters
# # infnity norm if needed
# dist = y_l-t_l
# loss_all = torch.linalg.norm(dist, float('inf'))
# torch.max(loss_all)
# save the weights to calculate the Lipschitz constant
# save the fNN
# torch.save(fnet.cpu(), 'PF_fnet.pt')
f_w1 = fnet.layer1.weight.data.cpu().numpy()
f_w2 = fnet.layer2.weight.data.cpu().numpy()

np.savetxt("fw1.txt", f_w1, fmt="%s")
np.savetxt("fw2.txt", f_w2, fmt="%s")
# load fNN from a file
r = 3 # region of interest
N_l = 1500 # change according to delta
xl = np.linspace(-r,r,N_l, dtype = float)
xl2 = np.linspace(-r,r,N_l, dtype = float)
xl3 = np.linspace(-r,r,N_l, dtype = float)
xl4 = np.linspace(-r,r,N_l, dtype = float)

u_bdd = 5.
v_bdd = 1.  # bound for input
# check the rough value of alpha
x_l = []
x_l2 = []
x_l3 = []
x_l4 = []

N_l = 1500 
for i in range(0,N_l): 
    for j in range(0,N_l):
        x_l.append([xl[j],xl[i]])

x_l = torch.tensor(x_l)
x_l = x_l.float()

for i in range(0,N_l): 
    for j in range(0,N_l):
        x_l2.append([xl2[j],xl2[i]])

x_l2 = torch.tensor(x_l2)
x_l2 = x_l2.float()

for i in range(0,N_l): 
    for j in range(0,N_l):
        x_l3.append([xl3[j],xl3[i]])

x_l3 = torch.tensor(x_l3)
x_l3 = x_l3.float()

for i in range(0,N_l): 
    for j in range(0,N_l):
        x_l4.append([xl4[j],xl4[i]])

x_l4 = torch.tensor(x_l4)
x_l4 = x_l4.float()

u_l = torch.Tensor(len(x_l), 1).uniform_(-u_bdd, u_bdd)
v_l = torch.Tensor(len(x_l), 1).uniform_(-v_bdd, v_bdd)  

# fnet = fnet.to(device) # if need to switch device
#y_l = fnet(x_bdd)

# maximum of loss
#loss_all = torch.norm(y_l-t_l1, dim = 1)
#alpha = torch.max(loss_all)

#f_w1 = fnet.layer1.weight.data.cpu().numpy()
#f_w2 = fnet.layer2.weight.data.cpu().numpy()

#print(alpha)

# fnet = torch.load('PF_fnet.pt').to(device)

def f_learned1(x1,x2,x3,x4,u,v):
    X = torch.cat((x1,x2,x3,x4,u,v),1)
    y1,y2,y3,y4 = fnet(X)
    return y1,y2,y3,y4

def f_learned(x1,x2,x3,x4,u):
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    v = []
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    y1,y2,y3,y4 = f_learned1(x1,x2,x3,x4,u,v)
    for r in range(0, len(x1)):

        v1[:, 0] = (x1[r][1] * np.sin(x2[r][0]) + x1[r][0] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][0]))
        v2[:, 0] = (x2[r][0] + x2[r][1] * np.cos(x1[r][1]**2) + x2[r][1] * np.sin(x3[r][1]))
        v3[:, 0] = (x3[r][0] * np.cos(x1[r][0]) + x3[r][1] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][1]))
        v4[:, 0] = (x4[r][0] * np.sin(x3[r][0]) + x4[r][1] * np.cos(x2[r][0]))
        y1[:, 1] = y1[:, 1] + (2 - np.cos(x1[r][0])**2) * (v1[:, 0]+u[:, 0])
        y2[:, 1] = y2[:, 1] + (1 + np.cos(x2[r][1])) * (v2[:, 0]+u[:, 0])
        y3[:, 1] = y3[:, 1] + (0.5 * x3[r][1] * (2 + np.cos(x3[r][0])**2)**2 + 2) * (v3[:, 0]+u[:, 0])
        y4[:, 1] = y4[:, 1] + (v4[:, 0]+u[:, 0])

    return y1,y2,y3,y4

def f_learn(x1,x2,x3,x4,u):
    y1 = []
    y2 = []
    y3 = []
    y4 = []

    for r in range(0, len(x1)):
    # Define the system dynamics
       
        f1 = [-x1[r][1], 
              x1[r][0] * x1[r][1] + (2 - np.cos(x1[r][0])**2) * (x1[r][1] * np.sin(x2[r][0]) + x1[r][0] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][0]))]
        y1.append(f1)

        f2 = [x2[r][1], 
              -x2[r][0] - 0.5 * x2[r][1] + 0.5 * x2[r][0]**2 * x2[r][1] + (1 + np.cos(x2[r][1])) * (x2[r][0] + x2[r][1] * np.cos(x1[r][1]**2) + x2[r][1] * np.sin(x3[r][1]))]
        y2.append(f2)

        f3 = [-x3[r][0] + x3[r][1], 
              -0.5 * (x3[r][0] + x3[r][1]) + (0.5 * x3[r][1] * (2 + np.cos(x3[r][0])**2)**2 + 2)* (x3[r][0] * np.cos(x1[r][0]) + x3[r][1] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][1]))]
        y3.append(f3)

        f4 = [x4[r][1], 
              -x4[r][0] - 8 * x4[r][1] + (x4[r][0] * np.sin(x3[r][0]) + x4[r][1] * np.cos(x2[r][0]))]
        y4.append(f4)
 
    y1 = torch.tensor(y1)
    y2 = torch.tensor(y2)
    y3 = torch.tensor(y3)
    y4 = torch.tensor(y4)
    y1[:, 1] = y1[:, 1] + (2 - np.cos(x1[r][0])**2) * u[:, 0]
    y2[:, 1] = y2[:, 1] + (1 + np.cos(x2[r][1])) * u[:, 0]
    y3[:, 1] = y3[:, 1] + (0.5 * x3[r][1] * (2 + np.cos(x3[r][0])**2)**2 + 2) * u[:, 0]
    y4[:, 1] = y4[:, 1] + u[:, 0]
    
    return y1, y2, y3, y4

def V(x1, x2, x3, x4):
    V1 = []
    V2 = []
    V3 = []
    V4 = []

    for r in range(0, len(x1)):
        v1_val = (x1[r][1] * np.sin(x2[r][0]) + x1[r][0] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][0]))
        V1.append([v1_val])

        v2_val = (x2[r][0] + x2[r][1] * np.cos(x1[r][1]**2) + x2[r][1] * np.sin(x3[r][1]))
        V2.append([v2_val])

        v3_val = (x3[r][0] * np.cos(x1[r][0]) + x3[r][1] * np.sin(x2[r][0]**2) + x1[r][1] * np.cos(x3[r][1]))
        V3.append([v3_val])

        v4_val = (x4[r][0] * np.sin(x3[r][0]) + x4[r][1] * np.cos(x2[r][0]))
        V4.append([v4_val])

    V1 = torch.tensor(V1)
    V2 = torch.tensor(V2)
    V3 = torch.tensor(V3)
    V4 = torch.tensor(V4)

    return V1, V2, V3, V4

class L_Net(torch.nn.Module):

    def __init__(self):
        super(L_Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = nn.Linear(2, 6)
        self.layer2 = nn.Linear(6, 1)
        self.layer1_2 = nn.Linear(2, 6)
        self.layer2_2 = nn.Linear(6, 1)
        self.layer1_3 = nn.Linear(2, 6)
        self.layer2_3 = nn.Linear(6, 1)
        self.layer1_4 = nn.Linear(2, 6)
        self.layer2_4 = nn.Linear(6, 1)
        self.control = nn.Linear(2, 1, bias=False)
        self.layer3_1 = nn.Linear(3, 12)  
        self.layer4_1 = nn.Linear(12, 1)
        self.layer3_2 = nn.Linear(3, 12)  
        self.layer4_2 = nn.Linear(12, 1)
        self.layer3_3 = nn.Linear(3, 12)  
        self.layer4_3 = nn.Linear(12, 1)
        self.layer3_4 = nn.Linear(3, 12)  
        self.layer4_4 = nn.Linear(12, 1)
        self.to(device)

    def forward(self, x_1, x_2, x_3, x_4,v1,v2,v3,v4):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x_1))
        out = sigmoid(self.layer2(h_1))
        h_1_2 = sigmoid(self.layer1(x_2))
        out_2 = sigmoid(self.layer2(h_1_2))
        h_1_3 = sigmoid(self.layer1(x_3))
        out_3 = sigmoid(self.layer2(h_1_3))
        h_1_4 = sigmoid(self.layer1(x_4))
        out_4 = sigmoid(self.layer2(h_1_4))
        u = self.control(x_1)
        combined_input1 = torch.cat((x_1, v1), dim=1)
        combined_input2 = torch.cat((x_2, v2), dim=1)
        combined_input3 = torch.cat((x_3, v3), dim=1)
        combined_input4 = torch.cat((x_4, v4), dim=1)
        h_21 = sigmoid(self.layer3_1(combined_input1))
        out21 = sigmoid(self.layer4_1(h_21))
        h_22 = sigmoid(self.layer3_2(combined_input2))
        out22 = sigmoid(self.layer4_2(h_22))
        h_23 = sigmoid(self.layer3_3(combined_input3))
        out23 = sigmoid(self.layer4_3(h_23))
        h_24 = sigmoid(self.layer3_4(combined_input4))
        out24 = sigmoid(self.layer4_4(h_24))
        return out,out_2,out_3,out_4, u, out21, out22, out23, out24


def CheckLyapunov(x, f, V,RO ,ball_lb, ball_ub, config, epsilon1,epsilon2,epsilon3):    
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
                           logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon2),
                           logical_imply(ball_in_bound, RO<0))
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

def CheckdVdx(x, V, ball_ub, config, M):    
    # Given a candidate Lyapunov function V, check the Lipschitz constant within a domain around the origin (sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    ball= Expression(0)
    derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        derivative_of_V += V.Differentiate(x[i])*V.Differentiate(x[i])
    ball_in_bound = logical_and(ball <= ball_ub*ball_ub)
    
    # Constraint: x ∈ Ball → partial derivative of V <= M     
    condition = logical_imply(ball_in_bound, derivative_of_V <= M)
    return CheckSatisfiability(logical_not(condition),config)

N = 800  # sample size
c_1 = 1
c_2 = 40
c_3 = 3
torch.manual_seed(10)
x1 = torch.Tensor(N, 2).uniform_(-6, 6)
x2 = torch.Tensor(N, 2).uniform_(-6, 6)
x3 = torch.Tensor(N, 2).uniform_(-6, 6)
x4 = torch.Tensor(N, 2).uniform_(-6, 6)

x_0 = torch.zeros([1, 2])
x_02 = torch.zeros([1, 2])

x_0 = x_0.to(device)
x_02 = x_02.to(device)
X1 = Variable("X1")
X2 = Variable("X2")
X3 = Variable("X3")
X4 = Variable("X4")
X5 = Variable("X5")
X6 = Variable("X6")
X7 = Variable("X7")
X8 = Variable("X8")

vars_ = [X1,X2]
vars_1 = [X1,X2,X3]
vars_2 = [X3,X4,X5]
vars_3 = [X5,X6,X7]
vars_4 = [X7,X8,X6]
config = Config()
config.use_polytope_in_forall = True
config.use_local_optimization = True
config.precision = 1e-2
epsilon1 = torch.mean( c_1 *torch.norm(x1, dim=1))
epsilon2 = torch.mean( c_2 *torch.norm(x1, dim=1))
epsilon3 = torch.mean( c_3 *torch.norm(x1, dim=1))
# Checking candidate V within a ball around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub)
ball_lb = 0.2
ball_ub = 3
beta = -0.05
Kf = 1.48
KF = 2.09
d = 1e-6
loss = 0.004

out_iters = 0
valid = False

while out_iters < 2 and not valid: 
    start = timeit.default_timer()
    model = L_Net()
    L = []
    i = 0 
    t = 0
    max_iters = 2000
    learning_rate = 0.06
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #f_w1 = fnet.layer1.weight.data.cpu().numpy()
    #f_w2 = fnet.layer2.weight.data.cpu().numpy()
    #f_b1 = fnet.layer1.bias.data.cpu().numpy()
    #f_b2 = fnet.layer2.bias.data.cpu().numpy()

    while i < max_iters and not valid:

        V1,V2,V3,V4 = V(x1.cpu(), x2.cpu(), x3.cpu(), x4.cpu())
        V1 = V1.float()
        V2 = V2.float()
        V3 = V3.float()
        V4 = V4.float()
        V1 = V1.to(device)
        V2 = V2.to(device)
        V3 = V3.to(device)
        V4 = V4.to(device)
        x1 = x1.float()
        x1 = x1.to(device)
        x2 = x2.float()
        x2 = x2.to(device)
        x3 = x3.float()
        x3 = x3.to(device)
        x4 = x4.float()
        x4 = x4.to(device)
        v_candidate,v_candidate2,v_candidate3,v_candidate4,u,Ro,Ro2,Ro3,Ro4=model(x1,x2,x3,x4,V1,V2,V3,V4)
        f1, f2, f3, f4 = f_learn(x1.cpu(), x2.cpu(), x3.cpu(), x4.cpu(),u.cpu())
        f1 = f1.to(device)
        f2 = f2.to(device)
        f3 = f3.to(device)
        f4 = f4.to(device)
        x1_norm = torch.norm(x1, p=2, dim=1, keepdim=True)
        x2_norm = torch.norm(x2, p=2, dim=1, keepdim=True)
        x3_norm = torch.norm(x3, p=2, dim=1, keepdim=True)
        x4_norm = torch.norm(x4, p=2, dim=1, keepdim=True)
        Circle_Tuning = Tune(x1)
        Circle_Tuning = Circle_Tuning.to(device)
        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
        L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(v_candidate),model.layer2.weight)\
                            *dtanh(torch.tanh(torch.mm(x1,model.layer1.weight.t())+model.layer1.bias)),model.layer1.weight),f1.t()),0)
        L_V2 = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(v_candidate2),model.layer2_2.weight)\
                            *dtanh(torch.tanh(torch.mm(x2,model.layer1_2.weight.t())+model.layer1_2.bias)),model.layer1_2.weight),f2.t()),0)
        L_V3 = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(v_candidate3),model.layer2_3.weight)\
                            *dtanh(torch.tanh(torch.mm(x3,model.layer1_3.weight.t())+model.layer1_3.bias)),model.layer1_3.weight),f3.t()),0)
        L_V4 = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(v_candidate4),model.layer2_4.weight)\
                            *dtanh(torch.tanh(torch.mm(x4,model.layer1_4.weight.t())+model.layer1_4.bias)),model.layer1_4.weight),f4.t()),0)
        dVdx = torch.mm(torch.mm(dtanh(v_candidate),model.layer2.weight)\
                            *dtanh(torch.tanh(torch.mm(x1,model.layer1.weight.t())+model.layer1.bias)),model.layer1.weight)
        # With tuning term 
        loss1 = (relu(-v_candidate + c_1 * x1_norm) + relu(v_candidate - c_2 * x1_norm) + relu(
           L_V + c_3 * x1_norm - Ro)).mean()+1.2*((Circle_Tuning-6*v_candidate).pow(2)).mean()
        loss2 = (relu(-v_candidate2 + c_1 * x2_norm) + relu(v_candidate2 - c_2 * x2_norm) + relu(
           L_V + c_3 * x2_norm - Ro2)).mean()+1.2*((Circle_Tuning-6*v_candidate2).pow(2)).mean()
        loss3 = (relu(-v_candidate3 + c_1 * x3_norm) + relu(v_candidate3 - c_2 * x3_norm) + relu(
           L_V + c_3 * x3_norm - Ro3)).mean()+1.2*((Circle_Tuning-6*v_candidate3).pow(2)).mean()
        loss4 = (relu(-v_candidate4 + c_1 * x4_norm) + relu(v_candidate4 - c_2 * x4_norm) + relu(
           L_V + c_3 * x4_norm - Ro4)).mean()+1.2*((Circle_Tuning-6*v_candidate4).pow(2)).mean()
        loss = loss1+loss2+loss3+loss4 + 1.5*relu(Ro+Ro2+Ro3+Ro4).mean()
        
        print(i, "loss=",loss.item()) 
        L.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        w1 = model.layer1.weight.data.cpu().numpy()
        w2 = model.layer2.weight.data.cpu().numpy()
        B1 = model.layer1.bias.data.cpu().numpy()
        B2 = model.layer2.bias.data.cpu().numpy()
        w1_1 = model.layer3_1.weight.data.cpu().numpy()
        w2_1 = model.layer4_1.weight.data.cpu().numpy()
        B1_1 = model.layer3_1.bias.data.cpu().numpy()
        B2_1 = model.layer4_1.bias.data.cpu().numpy()
        w1_2 = model.layer3_2.weight.data.cpu().numpy()
        w2_2 = model.layer4_2.weight.data.cpu().numpy()
        B1_2 = model.layer3_2.bias.data.cpu().numpy()
        B2_2 = model.layer4_2.bias.data.cpu().numpy()
        w1_3 = model.layer3_3.weight.data.cpu().numpy()
        w2_3 = model.layer4_3.weight.data.cpu().numpy()
        B1_3 = model.layer3_3.bias.data.cpu().numpy()
        B2_3 = model.layer4_3.bias.data.cpu().numpy()
        w1_4 = model.layer3_4.weight.data.cpu().numpy()
        w2_4 = model.layer4_4.weight.data.cpu().numpy()
        B1_4 = model.layer3_4.bias.data.cpu().numpy()
        B2_4 = model.layer4_4.bias.data.cpu().numpy()
        q = model.control.weight.data.cpu().numpy()

        # Falsification
        if i % 10 == 0:
            u_NN = (q.item(0)*X1 + q.item(1)*X2) 
            print(u_NN)
            p1 = X2 * sin(X3) + X1 * sin(X3**2) + X2 * cos(X5)
            p2 = X3 + X4 * cos(X2**2) + X4 * sin(X6)
            p3 = X5 * cos(X1) + X6 * sin(X3**2) + X2 * cos(X6)
            p4 = X7 * sin(X5) + X8 * cos(X3)
            f1 = [-X2, (X1 * X2 + 2 - cos(X2)**2) * (u_NN + p1)]
            f2 = [X4, (-X3 - 0.5 * X4 + 0.5 * X3**2 * X4 + 1 + cos(X3)) * (p2)]
            f3 = [-X5 + X6, (-0.5 * (X5 + X6) + 0.5 * X6 * (2 + cos(X5)**2)**2 + 2) * (p3)]
            f4 = [X8, (-X7 - 8 * X8 + 1) * (p4)]

            z1 = np.dot(vars_,w1.T)+B1
            v1 = []
            for j in range(0,len(z1)):
                v1.append(tanh(z1[j]))
            z2 = np.dot(v1,w2.T)+B2

            z3 = np.dot(vars_1,w1_1.T)+B1_1
            v2 = []
            for j in range(0,len(z3)):
                v2.append(tanh(z3[j]))
            z4 = np.dot(v2,w2_1.T)+B2_1

            z5 = np.dot(vars_2,w1_2.T)+B1_2
            v3 = []
            for j in range(0,len(z5)):
                v3.append(tanh(z5[j]))
            z6 = np.dot(v3,w2_2.T)+B2_2

            z7 = np.dot(vars_3,w1_2.T)+B1_2
            v4 = []
            for j in range(0,len(z7)):
                v4.append(tanh(z7[j]))
            z8 = np.dot(v4,w2_3.T)+B2_3

            z9 = np.dot(vars_4,w1_4.T)+B1_4
            v5 = []
            for j in range(0,len(z9)):
                v5.append(tanh(z9[j]))
            z10 = np.dot(v5,w2_4.T)+B2_4

            V_learn = tanh(z2.item(0)) 
            RO1 = tanh(z4.item(0))
            RO2 = tanh(z6.item(0))
            RO3 = tanh(z8.item(0))
            RO4 = tanh(z10.item(0)) 
            RO = RO1+RO2+RO3+RO4

            print('===========Verifying==========')        
            start_ = timeit.default_timer()
            V1_1 =  torch.mean(torch.norm(V1, dim=1)) 
            result= CheckLyapunov(vars_, f1, V_learn, RO, ball_lb, ball_ub, config,epsilon1, epsilon2 + V1_1- beta, epsilon3 )
            stop_ = timeit.default_timer() 

            if (result): 
                print("Not a Lyapunov function. Found counterexample: ")
                print(result)
                x1 = x1.to('cpu')
                x2 = x2.to('cpu')
                x3 = x3.to('cpu')
                x4 = x4.to('cpu')
                x1 = AddCounterexamples(x1,result,10)
                x2 = AddCounterexamples(x2,result,10)
                x3 = AddCounterexamples(x3,result,10)
                x4 = AddCounterexamples(x4,result,10)
                
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
                result_strict= CheckLyapunov(vars_, f1, V_learn, ball_lb, ball_ub, config,epsilon1,epsilon2+V1-beta, epsilon3) # SMT solver
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