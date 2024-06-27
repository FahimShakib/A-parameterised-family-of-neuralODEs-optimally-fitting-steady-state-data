import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# import os

# from torchdiffeq import odeint_adjoint as odeint

# Contractive NodeREN with parameters to learn
class _System_contractive(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, alpha = 0.0, linear_output=False):
        """Used by the upper class NODE_REN to guarantee contractivity to the model. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 1.0           #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        # self.Ptilde = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.D22 = nn.Parameter(torch.randn(ny,nu,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        #Initialization of the last Parameters:
        self.A = torch.zeros(nx,nx,device=device)
        # self.Y= torch.zeros(nx,nx)
        self.D11 = torch.zeros(nq,nq,device=device) 
        self.C1 = torch.zeros(nq,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.P = torch.zeros(nx,nx,device=device)
        self.alpha= alpha 
        self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        """Used at the end of each batch training for the update of the constrained matrices.
        """
        P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device)
        # P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device) + self.Ptilde - self.Ptilde
        self.P = P
        H = F.linear(self.X,self.X) + self.epsilon*torch.eye(self.nx+self.nq,device=self.device)
        #Partition of H in --> [H1 H2;H3 H4]
        h1,h2 = torch.split(H, (self.nx,self.nq), dim =0) # you split the matrices in two big rows
        H1, H2 = torch.split(h1, (self.nx,self.nq), dim=1) # you split each big row in two chunks
        H3, H4 = torch.split(h2, (self.nx,self.nq), dim=1)

        Y= -0.5*(H1+ self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P),Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda),torch.tril(H4,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda),self.Chi)
        Z = -H2-self.Chi
        self.B1 = F.linear(torch.inverse(P),Z.T)        

    def forward(self,t,xi,u):
        n_initial_states = xi.shape[0]
        # By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        # v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + u*self.D12[0,:].T ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            # v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + u*self.D12[i,:].T ).unsqueeze(1)
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            # + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        # xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1,device=self.device),self.bx) + u*self.B2.T
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1,device=self.device),self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_#,yi

    def output(self,xi, u ):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        # v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ u*self.D12[0,:].T  ).unsqueeze(1)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ F.linear(u, self.D12[0,:])  ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt

    def calculate_w(self,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        return w

# Class used for the signal generator - no parameters in it
class _System_contractive_no_pars(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, alpha = 0.0, linear_output=False):
        """Used by the upper class NODE_REN to guarantee contractivity to the model. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 1.0           #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        self.Pstar = (torch.randn(nx,nx,device=device)*std)
        # self.Ptilde = (torch.randn(nx,nx,device=device)*std)
        self.Chi = (torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        self.Y1 = (torch.randn(nx,nx,device=device)*std)
        self.B2 = (torch.randn(nx,nu,device=device)*std)
        self.D12 = (torch.randn(nq,nu,device=device)*std)
        self.C2 = (torch.randn(ny,nx,device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = (torch.randn(ny,nq,device=device)*std)
        self.D22 = (torch.randn(ny,nu,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= (torch.randn(nx,1,device=device)*std)
            self.bv= (torch.randn(nq,1,device=device)*std)
            self.by= (torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        self.X = (torch.randn(nx+nq,nx+nq,device=device)*std)
        #Initialization of the last Parameters:
        self.A = torch.zeros(nx,nx,device=device)
        # self.Y= torch.zeros(nx,nx)
        self.D11 = torch.zeros(nq,nq,device=device) 
        self.C1 = torch.zeros(nq,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.P = torch.zeros(nx,nx,device=device)
        self.alpha= alpha 
        self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        """Used at the end of each batch training for the update of the constrained matrices.
        """
        P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device)
        # P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device) + self.Ptilde - self.Ptilde
        self.P = P
        H = F.linear(self.X,self.X) + self.epsilon*torch.eye(self.nx+self.nq,device=self.device)
        #Partition of H in --> [H1 H2;H3 H4]
        h1,h2 = torch.split(H, (self.nx,self.nq), dim =0) # you split the matrices in two big rows
        H1, H2 = torch.split(h1, (self.nx,self.nq), dim=1) # you split each big row in two chunks
        H3, H4 = torch.split(h2, (self.nx,self.nq), dim=1)

        Y= -0.5*(H1+ self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P),Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda),torch.tril(H4,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda),self.Chi)
        Z = -H2-self.Chi
        self.B1 = F.linear(torch.inverse(P),Z.T)        

    def forward(self,t,xi,u):
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        # v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + u*self.D12[0,:].T ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            # v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + u*self.D12[i,:].T ).unsqueeze(1)
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
    
            w = w + F.linear(self.act(v),vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1,device=self.device),self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_#,yi

    def output(self,xi, u ):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        # v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ u*self.D12[0,:].T  ).unsqueeze(1)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ F.linear(u,self.D12[0,:])  ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            # v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + u*self.D12[i,:].T ).unsqueeze(1)
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u,self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        # yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + u*self.D22.T + By
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u,self.D22) + By
        return yt

    def calculate_w(self,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        return w
    
# Class used to initialize the model
class _System_general(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, linear_output=False):
        """Used by the upper class NODE_REN. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 1.0         #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        # self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        # self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        # self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.A = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.B1 = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.C1 = nn.Parameter(torch.randn(nq,nx,device=device)*std)
        # self.D11_coefficients = nn.Parameter(torch.randn(nq,device=device)*std) 
        self.D11 = nn.Parameter(torch.randn(nq,nq,device=device)*std) 
        self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.D22 = nn.Parameter(torch.randn(ny,nu,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        # self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        #Initialization of the last Parameters:
        # self.Y= torch.zeros(nx,nx)
        # self.P = torch.zeros(nx,nx,device=device)
        # self.alpha= alpha 
        # self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()     

    def forward(self, t, xi, u):
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        # v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + u*self.D12[0,:].T).unsqueeze(1)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + F.linear(u,self.D12[0,:]) ).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            # v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + u*self.D12[i, :].T).unsqueeze(1)
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        # xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states, 1, device=self.device), self.bx) + u*self.B2
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states, 1, device=self.device), self.bx) + F.linear(u,self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_  # ,yi
    def updateParameters(self):
        #A general NodeREN does not require any additional step.
        pass
    def output(self, xi, u):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        # v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + u*self.D12[0,:].T).unsqueeze(1)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + F.linear(u,self.D12[0,:]) ).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            # v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + u*self.D12[i, :].T).unsqueeze(1)
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + F.linear(u,self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        # yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + u*self.D22.T + By
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u,self.D22) + By
        return yt

    def calculate_w(self,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        return w

# Class used to initialize the models that achieve moment matching
class _System_general_moment_matching(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, linear_output=False):
        """Used by the upper class NODE_REN. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 1.0         #standard deviation used to draw randomly the initial weights of the model.
        # Moment matching parameters B2, D12, D22
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.D22 = nn.Parameter(torch.randn(ny,nu,device=device)*std)
        
        self.A = torch.randn(nx,nx,device=device)*std
        self.B1 = torch.randn(nx,nq,device=device)*std
        
        self.C1 = torch.randn(nq,nx,device=device)*std
        self.D11 = torch.randn(nq,nq,device=device)*std
        self.C2 = torch.randn(ny,nx,device=device)*std
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = torch.randn(ny,nq,device=device)*std
        BIAS = bias
        if(BIAS):
            self.bx= torch.randn(nx,1,device=device)*std
            self.bv= torch.randn(nq,1,device=device)*std
            self.by= torch.randn(ny,1,device=device)*std
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        # self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()     

    def forward(self, t, xi, u):
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        # v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + u*self.D12[0,:].T).unsqueeze(1)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + F.linear(u,self.D12[0,:]) ).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            # v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + u*self.D12[i, :].T).unsqueeze(1)
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        # xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states, 1, device=self.device), self.bx) + u*self.B2
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states, 1, device=self.device), self.bx) + F.linear(u,self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_  # ,yi
    def updateParameters(self,SG,Z,YY,beta):
        """Used at the end of each batch training for the update of the constrained matrices.
        """
        self.A.data  = SG.sys.A.data - torch.matmul(self.B2.data,SG.sys.C2.data)
        self.B1.data = SG.sys.B1.data - torch.matmul(self.B2.data,SG.sys.D21.data)
        self.bx.data = SG.sys.bx.data - torch.matmul(self.B2.data,SG.sys.by.data)
    
        # D12 (nq,nu) is a 'free variable' - it is used to enforce additional properties
        self.C1.data  = SG.sys.C1.data - torch.matmul(self.D12.data,SG.sys.C2.data)
        self.D11.data = SG.sys.D11.data - torch.matmul(self.D12.data,SG.sys.D21.data)
        self.bv.data  = SG.sys.bv.data - torch.matmul(self.D12.data,SG.sys.by.data)
    
        # To match the output equation
        # D22 (ny,nu) is a 'free variable'
        self.C2.data  = torch.Tensor(Z) - torch.matmul(self.D22.data,SG.sys.C2.data)
        self.D21.data = torch.Tensor([YY]) - torch.matmul(self.D22.data,SG.sys.D21.data)
        self.by.data  = torch.Tensor([beta]) - torch.matmul(self.D22.data,SG.sys.by.data)
        
    def output(self, xi, u):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        # v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + u*self.D12[0,:].T).unsqueeze(1)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + F.linear(u,self.D12[0,:]) ).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            # v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + u*self.D12[i, :].T).unsqueeze(1)
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,device=self.device) + F.linear(u,self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        # yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + u*self.D22.T + By
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u,self.D22) + By
        return yt

    def calculate_w(self,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        return w

# Class used to initialize the model
class _System_general_static(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, linear_output=False):
        """Used by the upper class NODE_REN. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 1.0         #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        # self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        # self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        # self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.A = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.B1 = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.C1 = nn.Parameter(torch.randn(nq,nx,device=device)*std)
        # self.D11_coefficients = nn.Parameter(torch.randn(nq,device=device)*std) 
        self.D11 = nn.Parameter(torch.randn(nq,nq,device=device)*std) 
        self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.D22 = nn.Parameter(torch.randn(ny,nu,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()     

    def updateParameters(self):
        #A general NodeREN does not require any additional step.
        pass
    # The output function becomes the forward function
    def forward(self, t, xi, u):
        """Calculates the output yt given the state  and the input u.
        """
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(1, self.nq, device=self.device)
        v = (self.bv[0] + F.linear(u,self.D12[0,:]) ).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] + F.linear(u,self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        yt = F.linear(w, self.D21) + F.linear(u,self.D22) + self.by
        return yt

    def calculate_w(self,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        return w

        
class NODE_REN(nn.Module):
    def __init__(self, nx = 5, ny = 5, nu = 5, nq = 5, sigma = "tanh", epsilon = 1.0e-2, mode = "c", gamma = 1., device = "cpu", bias = False, ni = 1., rho = 1., alpha = 0.0, linear_output=False, u=0):
        """Base class for Neural Ordinary Differential Equation Recurrent Equilbrium Networks (NODE_RENs).
        
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'. Defaults to "tanh".
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. Defaults to 1.0e-2.
            -mode (str, optional): Property to ensure. Possible options: 'c'= contractive model, 'rl2'=L2 lipschitz-bounded, 'input_p'=input passive model, 'output_p'=output_passive model.
            -gamma (float, optional): If the model is L2 lipschitz bounded (i.e., mode == 'c'), gamma is the L2 Lipschitz constant. Defaults to 1.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -ni  (float, optional): If the model is input passive (i.e., mode == 'input_p') , ni is the weight coefficient that characterizes the (input passive) supply rate function.
            -rho (float, optional): If the model is output passive (i.e., mode == 'output_p'), rho is the weight coefficient that characterizes the (output passive) supply rate function.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
            - u: external input function
        """
        super().__init__()
        self.mode = mode.lower()
        self.nfe = 0
        self.u = u
        if (self.mode == "c"):
            self.sys = _System_contractive(nx, ny, nu, nq,sigma, epsilon, device=device,bias=bias, linear_output=linear_output,alpha=alpha)
        elif (self.mode == "c_no_pars"):
            self.sys = _System_contractive_no_pars(nx, ny, nu, nq,sigma, epsilon, device=device,bias=bias, linear_output=linear_output,alpha=alpha)
        elif (self.mode == "general"):
            self.sys = _System_general(nx, ny, nu, nq,sigma, epsilon, device=device,bias=bias, linear_output=linear_output)
        elif (self.mode == "general_moment_matching"):
            self.sys = _System_general_moment_matching(nx, ny, nu, nq,sigma, epsilon, device=device,bias=bias, linear_output=linear_output)
        elif (self.mode == "general_static"):
            self.sys = _System_general_static(nx, ny, nu, nq,sigma, epsilon, device=device,bias=bias, linear_output=linear_output)
        else:
            raise NameError("The inserted mode is not valid. Please write 'c', 'c_no_pars', or 'general'.")

    def updateParameters(self):
        self.sys.updateParameters()

    def forward(self, t, x):
        self.nfe += 1
        u = self.u(torch.Tensor(t).detach().numpy())
        xdot = self.sys(t, x, u)
        # xdot = self.sys(t,x)
        return xdot

    def output(self,t,x):
        # u = self.u(t)
        u = self.u(torch.Tensor(t).detach().numpy())
        yt = self.sys.output(x,u)
        return yt
    @property
    def nfe(self):
        return self._nfe
    @nfe.setter
    def nfe(self,value):
        self._nfe = value
    
