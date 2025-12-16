import torch
import torch.utils.data.dataset as Dataset
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

torch.manual_seed(42)


housing = fetch_california_housing()
X_data = housing.data
y_data = housing.target.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)


scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

X = X_train
y = y_train




class Dense(torch.nn.Module):
    def __init__(self,input_features,Hidden_neurons,output_feature, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.in_features = input_features
        self.Layer_stack = []
        for i in Hidden_neurons:
            self.Layer_stack.append(torch.nn.Linear(self.in_features,i))
            self.Layer_stack.append(torch.nn.ReLU())
            self.in_features = i
        self.Layer_stack.append(torch.nn.Linear(self.in_features,out_features=output_feature))
        self.Model = torch.nn.Sequential(*self.Layer_stack)
    def forward(self,X):
        return self.Model(X)
    

class ADAM:
    def __init__(self,Model_params,lr = 0.001,beta1 = 0.9,beta2 = 0.999,*args,**kwargs):
        self.state = {}
        for i in Model_params.keys():
            self.state[i] = {"weights":
                        {"params":Model_params[i]["weights"],        
                        "mt":torch.zeros_like(Model_params[i]["weights"]),
                        "vt":torch.zeros_like(Model_params[i]["weights"]),
                    },

                "bias":{
                    "params":Model_params[i]["bias"],        
                    "mt":torch.zeros_like(Model_params[i]["bias"]),
                    "vt":torch.zeros_like(Model_params[i]["bias"]),
                 },
                    "step":0
            }
        self.initial_lr = lr
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8


    
    def step(self):
        for i in self.state.keys():
            with torch.no_grad():
                self.state[i]["step"] += 1
                t = self.state[i]["step"]
                '''weight Update of mt,vt for a layer'''
                grd_weights = self.state[i]["weights"]["params"].grad
                grd_bias = self.state[i]["bias"]["params"].grad

                self.state[i]["weights"]["mt"] = self.state[i]["weights"]["mt"]*self.beta1 + (1-self.beta1)*grd_weights
                self.state[i]["weights"]["vt"] = self.state[i]["weights"]["vt"]*self.beta2 + (1-self.beta2)*(grd_weights*grd_weights)
                '''bias update of mt,vt for a layer'''
                self.state[i]["bias"]["mt"] = self.state[i]["bias"]["mt"]*self.beta1 + (1-self.beta1)*grd_bias
                self.state[i]["bias"]["vt"] = self.state[i]["bias"]["vt"]*self.beta2 + (1-self.beta2)*(grd_bias*grd_bias)

                '''bias correction'''
                mt_ini_weights = self.state[i]["weights"]["mt"]/(1-self.beta1**t)
                vt_ini_weights  = self.state[i]["weights"]["vt"]/(1-self.beta2**t)
                mt_ini_bias = self.state[i]["bias"]["mt"]/((1-self.beta1**t))
                vt_ini_bias  = self.state[i]["bias"]["vt"]/(1-self.beta2**t)


                step_size_weigths = self.lr*((mt_ini_weights)/(torch.sqrt(vt_ini_weights) + self.eps))
                step_size_bias = self.lr*((mt_ini_bias)/(torch.sqrt(vt_ini_bias) + self.eps))
                
                '''weights descent/bias descent'''
                self.state[i]["weights"]["params"].data -= step_size_weigths
                self.state[i]["bias"]["params"].data -= step_size_bias
                # self.Lr_schedular(t)

  


    def zero_grad(self):
        for i in self.state.keys():
            if self.state[i]["weights"]["params"].grad is not None:
                self.state[i]["weights"]["params"].grad.zero_()
            if self.state[i]["bias"]["params"].grad is not None:
                self.state[i]["bias"]["params"].grad.zero_()


    def Lr_schedular(self,timestep,schedual_type = "expo"):
        if schedual_type == "step":
            drop = 700
            gamma = 0.75
            self.lr = self.initial_lr * (gamma ** (timestep // drop))
        elif schedual_type == "inverse":
            decay_rate = 0.0005
            self.lr = self.initial_lr / (1 + decay_rate * timestep)
        else:
            decay_rate = 0.001
            self.lr = self.initial_lr * np.exp(-decay_rate * timestep)


        
        


Model = Dense(X.shape[1],[64,32,16],1)
def generate_param_dict(stack):
    params = {}
    layer_idx = 0
    for i in range(len(stack)):
        if isinstance(stack[i], torch.nn.Linear):
            param = list(stack[i].parameters())
            params[f"Layer{layer_idx}"] = {"weights": param[0], "bias": param[1]}
            layer_idx += 1
    return params

params = generate_param_dict(Model.Layer_stack)

beta_1 = 0.9
beta_2 = 0.999
lr = 0.001
loss_list = []
optimizer = ADAM(Model_params=params,lr= lr,beta1=beta_1,beta2=beta_2)

loss_fn = torch.nn.MSELoss()
epocs = 3000
print("___Custom Adam___")
for i in range(epocs):
    y_pred = Model(X)
    loss = loss_fn(y_pred,y)

    if torch.isnan(loss):
        print(f"NaN detected at epoch {i}!")
        break

    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()


    if i%100 == 0: print(f"Loss->{loss.item()}")

    loss_list.append(loss.item())



Model = Dense(X.shape[1],[64,32,16],1)

print("___Torch Adam___")
optim = torch.optim.Adam(Model.parameters())
loss_list_torch = []
for i in range(epocs):
    y_pred = Model(X)
    loss = loss_fn(y_pred,y)

    if torch.isnan(loss):
        print(f"NaN detected at epoch {i}!")
        break

    optim.zero_grad()
    
    loss.backward()
    
    optim.step()


    if i%100 == 0: print(f"Loss->{loss.item()}")

    loss_list_torch.append(loss.item())


print("Loss Comparison Between Torch Adam and Custom Adam")

plt.figure(figsize=(15,15))
plt.plot(loss_list,'b',lw = 2,label = "Custom Adam Loss")
plt.plot(loss_list_torch,'r',lw = 2,label = "Torch Adam Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()

