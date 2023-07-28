import numpy as np
import torch
import math
from torchviz import make_dot

# Section to specify the settings of torch
dtype = torch.float64
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)
class simple_model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((1,1), dtype = dtype, device=device, requires_grad = True))
        self.b = torch.nn.Parameter(torch.randn((1,1), dtype = dtype, device=device, requires_grad = True))
        self.c = torch.nn.Parameter(torch.randn((1,1), dtype = dtype, device=device, requires_grad = True))
        self.d = torch.nn.Parameter(torch.randn((1,1), dtype = dtype, device=device, requires_grad = True))
    def forward(self, x):
        x = x.to(device)
        y_pred = self.a + self.b*x + self.c*(x**2) + self.d*(x**3)
        return y_pred

def main():
    # Variables tensors for the learning process
    a = torch.randn((), dtype = dtype, device=device, requires_grad = True)
    b = torch.randn((), dtype = dtype, device=device, requires_grad = True)
    c = torch.randn((), dtype = dtype, device=device, requires_grad = True)
    d = torch.randn((), dtype = dtype, device=device, requires_grad = True)

    # Desired Signal to aproximate
    t = torch.linspace(-math.pi, math.pi, 2000, dtype = dtype, device=device,)

    y = torch.sin(t)

    # Get cuda device

    # Deinition Learning rate
    learning_rate = torch.zeros((), dtype = dtype, requires_grad = True)
    lr = 1e-6

    # Model if the sytem class
    neural_network = simple_model()

    # Cost Function
    criterion = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.SGD(neural_network.parameters(), lr=1e-6)

    # Check graph 
    y_pred = neural_network(t)

    # Learning cycle
    for i in range(50000):
        # Prediction formulate
        y_pred = neural_network(t)

        # Generate error vector between the desired signal and the actual prediction
        loss = criterion(y_pred, y)

        print(i, loss.item())
        #loss.backward(retain_graph = True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #with torch.no_grad():
            #a.add_(-lr*a.grad)
            #b.add_(-lr*b.grad)
            #c.add_(-lr*c.grad)
            #d.add_(-lr*d.grad) 
    
            #a.grad.zero_()
            #b.grad.zero_()
            #c.grad.zero_()
            #d.grad.zero_()

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
    make_dot(y_pred, params = dict(neural_network.named_parameters())).render("nn", format="png")

if __name__ == '__main__':
    try:
        main()
        pass
    except(KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass