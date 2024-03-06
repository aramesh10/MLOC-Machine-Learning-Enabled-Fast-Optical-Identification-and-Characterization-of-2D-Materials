import torch

class OneDim_CNN(torch.nn.Module):
    
    def __init__(self):
        super(OneDim_CNN, self).__init__()

        self.conv1d_1 = torch.nn.Conv1d(100, 50, 10)
        self.activation_1 = torch.nn.ReLU()
        self.conv1d_2 = torch.nn.Conv1d(50, 20, 5)
        self.activation_2 = torch.nn.ReLU()
        self.linear_1 = torch.nn.Linear(20, 10)
        self.activation_3 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(10, 3)
        self.softmax = torch.nn.Softmax()
        
        # self.conv1d_3 = torch.nn.Conv1d(20, 100, 10)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):  
        x = self.conv1d_1(x)
        x = self.activation_1(x)
        x = self.conv1d_2(x)
        x = self.activation_2(x)
        x = self.linear_1(x)
        x = self.activation_3(x)
        x = self.linear_2(x)
        x = self.softmax(x)
        return x    
