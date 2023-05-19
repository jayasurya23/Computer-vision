import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        self.fc = nn.Linear(784, 100)               #input fc layer 28x28
        self.fc_out = nn.Linear(100, 10)
        self.conv1 = nn.Conv2d(1,40,5)
        self.conv2 = nn.Conv2d(40,40,5)
        self.conv_fc_in=nn.Linear(640,100)
        self.pooling=nn.MaxPool2d(2,2)

        self.fc_hidden1 = nn.Linear(100,100)


        #Layers needed for model 5
        self.fc_in_2 = nn.Linear(640,1000)
        self.fc_hidden_2 = nn.Linear(1000,1000)
        self.fc_out_2 = nn.Linear(1000,10)
        self.dropout = nn.Dropout(0.5)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
    
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.

        X = torch.flatten(X, 1)
        X = torch.sigmoid(self.fc(X))
        X = self.fc_out(X)
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        return  X
        # Delete line return NotImplementedError() once method is implemented.
        

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = self.pooling(torch.sigmoid(self.conv1(X)))
        X = self.pooling(torch.sigmoid(self.conv2(X)))                  #Convolutional layers with sigmoid activation function
        X=torch.flatten(X,1)
        X = torch.sigmoid(self.conv_fc_in(X))
        X = self.fc_out(X)
        # Uncomment the following return stmt once method implementation is done.
        # Delete line return NotImplementedError() once method is implemented.
        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.pooling(torch.relu(self.conv1(X)))
        X = self.pooling(torch.relu(self.conv2(X)))                     #Convolutional layers with ReLU activation function
        X=torch.flatten(X,1)
        X = torch.relu(self.conv_fc_in(X))
        X = self.fc_out(X)
        return X
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.pooling(torch.relu(self.conv1(X)))
        X = self.pooling(torch.relu(self.conv2(X)))
        X=torch.flatten(X,1)

        X = torch.relu(self.conv_fc_in(X))
        
        X = torch.relu(self.fc_hidden1(X))                                  #Extra fully connected layer 
        X = self.fc_out(X)
        return X
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.pooling(torch.relu(self.conv1(X)))
        X = self.pooling(torch.relu(self.conv2(X)))
        X = torch.flatten(X,1)
        X = torch.relu(self.fc_in_2(X))
        X = F.dropout(X)                                #Dropout of 0.5
        X = torch.relu(self.fc_hidden_2(X))
        X = F.dropout(X)
        X = self.fc_out_2(X)
        return X

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
    
    
