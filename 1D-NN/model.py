import torch

NUM_OF_CLASSES = 3

class OneDim_CNN(torch.nn.Module):
    
    def __init__(self):
        super(OneDim_CNN, self).__init__()

        # ENCODER
        self.conv1d_1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.activation_1 = torch.nn.ReLU()
        self.conv1d_2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)
        self.activation_2 = torch.nn.ReLU()

        self.max_pool_1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)              

        self.conv1d_3 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)               
        self.activation_3 = torch.nn.ReLU()
        self.conv1d_4 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)               
        self.activation_4 = torch.nn.ReLU()                

        self.max_pool_2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)                            
    

        self.conv1d_5 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)               
        self.activation_5 = torch.nn.ReLU()
        self.conv1d_6 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)               
        self.activation_6 = torch.nn.ReLU()   


        # DECODER
        self.up_conv_upsample_1 = torch.nn.Upsample(scale_factor=3)                                   
        self.up_conv_conv1d_1 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)     

        self.conv1d_7 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)              
        self.activation_7 = torch.nn.ReLU()
        self.conv1d_8 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)              
        self.activation_8 = torch.nn.ReLU()   

        self.up_conv_upsample_2 = torch.nn.Upsample(size=110)                                       
        self.up_conv_conv1d_2 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)     

        self.conv1d_9 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)              
        self.activation_9 = torch.nn.ReLU()
        self.conv1d_10 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)              
        self.activation_10 = torch.nn.ReLU()   

        self.conv1d_11 = torch.nn.Conv1d(in_channels=16, out_channels=3, kernel_size=1)              
        self.soft_max = torch.nn.Softmax(dim=1)

    def forward(self, x):  
        # ENCODER
        x = self.conv1d_1(x)            # L_IN = 100 | L_OUT = 96
        x = self.activation_1(x)        
        x = self.conv1d_2(x)            # L_IN = 96 | L_OUT = 92
        x = self.activation_2(x)

        x = self.max_pool_1(x)          # L_IN = 92 | L_OUT = 46

        x = self.conv1d_3(x)            # L_IN = 46 | L_OUT = 42
        x = self.activation_3(x)
        x = self.conv1d_4(x)            # L_IN = 42 | L_OUT = 38
        x = self.activation_4(x)

        x = self.max_pool_2(x)          # L_IN = 38 | L_OUT = 18


        x = self.conv1d_5(x)            # L_IN = 18 | L_OUT = 14
        x = self.activation_5(x)
        x = self.conv1d_6(x)            # L_IN = 14 | L_OUT = 10
        x = self.activation_6(x)


        # DECODER
        x = self.up_conv_upsample_1(x)  # L_IN = 10 | L_OUT = 30 .view(1,x.shape[0], x.shape[1])
        x = self.up_conv_conv1d_1(x)    # L_IN = 30 | L_OUT = 28 .view(x.shape[1], x.shape[2])

        # x = torch.concat((x, self.crop_arr(x, 28)))
        x = self.conv1d_7(x)            # L_IN = 28 | L_OUT = 24 
        x = self.activation_7(x)
        x = self.conv1d_8(x)            # L_IN = 24 | L_OUT = 20 
        x = self.activation_8(x)

        x = self.up_conv_upsample_2(x)  # L_IN = 20 | L_OUT = 110 .view(1,x.shape[0], x.shape[1])
        x = self.up_conv_conv1d_2(x)    # L_IN = 110 | L_OUT = 108 .view(x.shape[1], x.shape[2])

        x = self.conv1d_9(x)        # L_IN = 108 | L_OUT = 104 
        x = self.activation_9(x)
        x = self.conv1d_10(x)       # L_IN = 104 | L_OUT = 100
        x = self.activation_10(x)

        x = self.conv1d_11(x)       # L_IN = 100 | L_OUT = 100 
        x = self.soft_max(x)

        return x#.to(torch.float16)

    def crop_arr(self, arr, N):
        assert(len(arr) > N)

        length = len(arr)
        start_ind = 0 + ((length - N) // 2)
        end_ind = length - ((length - N) // 2)

        return arr[start_ind:end_ind]

class OneDim_CNN_2(torch.nn.Module):
    def __init__(self):
        super(OneDim_CNN_2, self).__init__()
        self.conv1d_1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.activation_1 = torch.nn.ReLU()
        self.conv1d_dropout_1 = torch.nn.Dropout1d(0.2)
        self.max_pool_1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)
        self.activation_2 = torch.nn.ReLU()
        self.conv1d_dropout_2 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=5)
        self.up_conv_upsample_2 = torch.nn.Upsample(size=104)
        self.conv1d_3 = torch.nn.Conv1d(in_channels=32, out_channels=3, kernel_size=5)
        self.soft_max = torch.nn.Softmax(dim=1)

    def forward(self, x):         
        x = self.conv1d_1(x)
        x = self.activation_1(x)
        x = self.conv1d_dropout_1(x)
        x = self.max_pool_1(x)
        x = self.conv1d_2(x)
        x = self.activation_2(x)
        x = self.conv1d_dropout_2(x)
        x = self.up_conv_upsample_2(x)
        x = self.conv1d_3(x)        
        x = self.soft_max(x)
        return x