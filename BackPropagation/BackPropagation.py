import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(42)
torch.manual_seed(42)

class Layer:
    def __init__(self,input_columns,number_of_neurons,Activation):
        self.input_columns = input_columns
        self.weights = np.random.randn(input_columns, number_of_neurons)* np.sqrt(2.0 / input_columns)
        self.bias = np.zeros((1,number_of_neurons))
        self.Relu = lambda x :  np.maximum(0,x)
        self.Activation = Activation
        self.dRelu = lambda x: np.where(x>0,1,0)

    # Forward pass Calculating the output
    def forward(self,X):
        self.input_ = X
        if self.Activation == True:
            self.z = np.matmul(X,self.weights) + self.bias
            return self.Relu(self.z)
        else:
            self.z = np.matmul(X,self.weights) + self.bias
            return self.z

    # Calulating and storing the Gradient of each Layer
    def backward(self,delta):
        drelu = self.dRelu(self.z)
        if self.Activation == True:
            self.dldw = np.matmul(self.input_.T,(delta*drelu))
            self.dldb = np.sum(delta * drelu, axis=0, keepdims=True) 
        else:
            self.dldw = np.matmul(self.input_.T,delta)
            self.dldb = np.sum(delta,axis=0,keepdims=True)
        return (delta * drelu) @ self.weights.T
    
    # Performing Grident descent
    def step(self,lr):
        self.weights -= lr*self.dldw
        self.bias -= lr*self.dldb

class MLP:
    def __init__(self):
        self.Layer_stack = []
        self.loss_list = []
        self.loss = lambda y,y_: (y_-y)**2
        self.dL = lambda y,y_ : 2*(y_ - y)
        self.output = []


    def add_layer(self,input_shape = 0,number_of_neurons = 1,Activation = False):
        if len(self.Layer_stack) == 0:
            self.Layer_stack.append(Layer(input_shape[1],number_of_neurons,Activation))
        else:
            input_shape = self.Layer_stack[-1].weights.shape[1]
            self.Layer_stack.append(Layer(input_shape,number_of_neurons,Activation))

    
    def forward(self,X):
        for i in range(len(self.Layer_stack)):
            X = self.Layer_stack[i].forward(X)
        return X
    

    def Backward(self,delta):
        for i in range(len(self.Layer_stack)-1,-1,-1):
            delta = self.Layer_stack[i].backward(delta)

    
    def step(self,lr):
        for i in range(len(self.Layer_stack)-1,-1,-1):
            self.Layer_stack[i].step(lr)



    def run(self,X,y,epoch = 10,lr = 0.001,batch_size=32):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        for ep in range(epoch):
            epoch_loss = 0
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process all batches in the epoch
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward Pass
                y_pred = self.forward(X_batch)
                
                # Calculating Loss
                l = np.mean(self.loss(y_batch, y_pred))
                epoch_loss += l
                
                # Calculate gradient
                delta = self.dL(y_batch, y_pred) / batch_size
                
                # Backward pass
                self.Backward(delta)
                
                # Performing Descent
                self.step(lr)
            
            # Store average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            self.loss_list.append(avg_epoch_loss)
            
            if(ep % 100 == 0):
                print(f"Epoch {ep}/{epoch}, Loss: {avg_epoch_loss:.6f}")
        
        self.output.append(self.forward(X))
        return self.loss_list, self.output

    
def generate_regression_data(n_samples=7000, n_features=4, noise=0.1, random_state=42, test_split=0.2):
    np.random.seed(random_state)
    
    # Generate features with different distributions
    X = np.random.randn(n_samples, n_features)
    
    # More complex non-linear function
    y = (
        2.0 * X[:, 0] +                    # Linear term
        3.0 * X[:, 1]**2 +                 # Quadratic term
        -1.5 * X[:, 0] * X[:, 2] +         # Interaction term
        0.8 * np.sin(2 * X[:, 1]) +        # Non-linear term
        0.5                                 # Bias
    )
    
    # Add Gaussian noise
    y += noise * np.random.randn(n_samples)
    y = y.reshape(-1, 1)
    
    # Train/test split
    n_train = int(n_samples * (1 - test_split))
    X_train = X[:n_train].astype(np.float32)
    X_test = X[n_train:].astype(np.float32)
    y_train = y[:n_train].astype(np.float32)
    y_test = y[n_train:].astype(np.float32)
    
    return X_train, X_test, y_train, y_test


class TORCH_MODEL(torch.nn.Module):
    def __init__(self,input_shape,Hidden_layers_sizes,output_size,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Layers = []
        self.name = ""
        self.pre = input_shape
        for i in Hidden_layers_sizes:
            l = torch.nn.Linear(self.pre,i)
            self.Layers.append(l)
            self.Layers.append(torch.nn.ReLU())
            self.pre = i
        self.Layers.append(torch.nn.Linear(self.pre,out_features=output_size))
        self.dense_model = torch.nn.Sequential(*self.Layers)

    def forward(self,X):
        return self.dense_model(X)

    def run(self,X,y,epochs,optim, loss_fn,lr =0.001,batch_size=32):
        loss_lst = []
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        for ep in range(epochs):
            epoch_loss = 0
            # Shuffle data
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                y_pred = self.forward(X_batch)
                loss = loss_fn(y_pred, y_batch)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / n_batches
            loss_lst.append(avg_epoch_loss)
            
            if ep % 100 == 0:
                print(f"Epoch {ep}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return loss_lst


def compare_models(custom_losses, pytorch_losses, custom_output, pytorch_output, y_true):
    """Compare Custom MLP and PyTorch model results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss Comparison
    axes[0, 0].plot(custom_losses, label='Custom MLP', alpha=0.7)
    axes[0, 0].plot(pytorch_losses, label='PyTorch', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final Predictions vs True Values (Custom MLP)
    custom_pred = custom_output[0].flatten()
    axes[0, 1].scatter(y_true.flatten(), custom_pred, alpha=0.5, s=10)
    axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Custom MLP: Predictions vs True')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final Predictions vs True Values (PyTorch)
    axes[1, 0].scatter(y_true.flatten(), pytorch_output.flatten(), alpha=0.5, s=10)
    axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('PyTorch: Predictions vs True')
    axes[1, 0].grid(True, alpha=0.3)
    
    #Residuals Comparison
    custom_residuals = y_true.flatten() - custom_pred
    pytorch_residuals = y_true.flatten() - pytorch_output.flatten()
    axes[1, 1].hist(custom_residuals, bins=50, alpha=0.5, label='Custom MLP')
    axes[1, 1].hist(pytorch_residuals, bins=50, alpha=0.5, label='PyTorch')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    custom_mse = np.mean((y_true.flatten() - custom_pred)**2)
    pytorch_mse = np.mean((y_true.flatten() - pytorch_output.flatten())**2)
    
    print(f"\nFinal MSE:")
    print(f"  Custom MLP:  {custom_mse:.6f}")
    print(f"  PyTorch:     {pytorch_mse:.6f}")
    print(f"  Difference:  {abs(custom_mse - pytorch_mse):.6f}")
    
    custom_r2 = 1 - (np.sum((y_true.flatten() - custom_pred)**2) / 
                     np.sum((y_true.flatten() - y_true.mean())**2))
    pytorch_r2 = 1 - (np.sum((y_true.flatten() - pytorch_output.flatten())**2) / 
                      np.sum((y_true.flatten() - y_true.mean())**2))
    
    print(f"\nR² Score:")
    print(f"  Custom MLP:  {custom_r2:.6f}")
    print(f"  PyTorch:     {pytorch_r2:.6f}")

    # Generate data
X_train, X_test, y_train, y_test = generate_regression_data(n_samples=1000, noise=0.1)
    
print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test data shape: X={X_test.shape}, y={y_test.shape}\n")

epochs = 1000
lr = 0.001
batch_size = 32
    

print("TRAINING CUSTOM MLP")

mlp = MLP()
mlp.add_layer(X_train.shape, 100, True)
mlp.add_layer(number_of_neurons=100, Activation=True)
mlp.add_layer(number_of_neurons=50, Activation=True)
mlp.add_layer(number_of_neurons=25, Activation=True)
mlp.add_layer(number_of_neurons=1)
    
custom_losses, custom_output = mlp.run(X_train, y_train, epoch=epochs, lr=lr, batch_size=batch_size)
    

print("TRAINING PYTORCH MODEL")

    
torch_model = TORCH_MODEL(
        input_shape=X_train.shape[1],
        Hidden_layers_sizes=[100, 100, 50, 25],
        output_size=1
    )
    
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    
optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()
    
pytorch_losses = torch_model.run(
        X_train_torch, y_train_torch, 
        epochs=epochs, 
        optim=optimizer, 
        loss_fn=loss_fn,
        batch_size=batch_size
    )
with torch.no_grad():
        pytorch_output = torch_model(X_train_torch).numpy()
    
compare_models(custom_losses, pytorch_losses, custom_output, pytorch_output, y_train)