import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Distance(nn.Module):
    def __init__(self):
        super(Distance, self).__init__()
        self.layer1 = nn.Linear(1, 8)
        self.layer2 = nn.Linear(8, 16)
        self.layer3 = nn.Linear(16, 32)
        self.layer4 = nn.Linear(32, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, 16)
        self.layer7 = nn.Linear(16, 8)
        self.layer8 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer5(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))
        x = self.layer8(x)
        return x
    
class Distance2(nn.Module):
    def __init__(self):
        super(Distance2, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 16)
        self.layer3 = nn.Linear(16, 32)
        self.layer4 = nn.Linear(32, 64)
        self.layer9 = nn.Linear(64, 128)
        self.layer10 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, 16)
        self.layer7 = nn.Linear(16, 8)
        self.layer8 = nn.Linear(8, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer5(self.activation(self.layer10(self.activation(self.layer9(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))))))
        x = self.layer8(x)
        return x

class Distance3(nn.Module):
    def __init__(self):
        super(Distance3, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 16)
        self.layer3 = nn.Linear(16, 32)
        self.layer4 = nn.Linear(32, 64)
        self.layer5 = nn.Linear(64, 128)
        self.layer11 = nn.Linear(128, 256)
        self.layer12 = nn.Linear(256, 128)
        self.layer6 = nn.Linear(128, 64)
        self.layer7 = nn.Linear(64, 32)
        self.layer8 = nn.Linear(32, 16)
        self.layer9 = nn.Linear(16, 8)
        self.layer10 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer12(self.activation(self.layer11(self.activation(self.layer5(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))))))))))
        x = self.layer10(x)
        return x
    
class Distance_Spherical(nn.Module):
    def __init__(self):
        super(Distance_Spherical, self).__init__()
        self.layer1 = nn.Linear(12, 24)
        self.layer2 = nn.Linear(24, 48)
        self.layer3 = nn.Linear(48, 96)
        self.layer4 = nn.Linear(96, 192)
        self.layer5 = nn.Linear(192, 96)
        self.layer6 = nn.Linear(96, 48)
        self.layer7 = nn.Linear(48, 24)
        self.layer8 = nn.Linear(24, 12)
        self.layer9 = nn.Linear(12, 4)
        self.layer10 = nn.Linear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer5(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))))))
        x = self.layer10(x)
        return x

class Distance_Spherical2(nn.Module):
    def __init__(self):
        super(Distance_Spherical, self).__init__()
        self.layer1 = nn.Linear(16, 24)
        self.layer2 = nn.Linear(24, 48)
        self.layer3 = nn.Linear(48, 96)
        self.layer6 = nn.Linear(96, 48)
        self.layer7 = nn.Linear(48, 24)
        self.layer8 = nn.Linear(24, 12)
        self.layer9 = nn.Linear(12, 4)
        self.layer10 = nn.Linear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))
        x = self.layer10(x)
        return x


    
class Angle1(nn.Module):
    def __init__(self):
        super(Angle1, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 16)
        self.layer3 = nn.Linear(16, 32)
        self.layer4 = nn.Linear(32, 64)
        self.layer5 = nn.Linear(64, 128)
        self.layer11 = nn.Linear(128, 256)
        self.layer12 = nn.Linear(256, 128)
        self.layer6 = nn.Linear(128, 64)
        self.layer7 = nn.Linear(64, 32)
        self.layer8 = nn.Linear(32, 16)
        self.layer9 = nn.Linear(16, 8)
        self.layer10 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer12(self.activation(self.layer11(self.activation(self.layer5(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))))))))))
        x = self.layer10(x)
        return x

class Angle2(nn.Module):
    def __init__(self):
        super(Angle2, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 16)
        self.layer3 = nn.Linear(16, 32)
        self.layer4 = nn.Linear(32, 64)
        self.layer5 = nn.Linear(64, 128)
        self.layer11 = nn.Linear(128, 256)
        self.layer12 = nn.Linear(256, 128)
        self.layer6 = nn.Linear(128, 64)
        self.layer7 = nn.Linear(64, 32)
        self.layer8 = nn.Linear(32, 16)
        self.layer9 = nn.Linear(16, 8)
        self.layer10 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer12(self.activation(self.layer11(self.activation(self.layer5(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))))))))))
        x = self.layer10(x)
        return x

class Angle3(nn.Module):
    def __init__(self):
        super(Angle3, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 16)
        self.layer3 = nn.Linear(16, 32)
        self.layer4 = nn.Linear(32, 64)
        self.layer5 = nn.Linear(64, 128)
        self.layer11 = nn.Linear(128, 256)
        self.layer12 = nn.Linear(256, 128)
        self.layer6 = nn.Linear(128, 64)
        self.layer7 = nn.Linear(64, 32)
        self.layer8 = nn.Linear(32, 16)
        self.layer9 = nn.Linear(16, 8)
        self.layer10 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer5(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))))))
        x = self.layer10(x)
        return x
    
class Angle_Spherical(nn.Module):
    def __init__(self):
        super(Angle_Spherical, self).__init__()
        self.layer1 = nn.Linear(12, 24)
        self.layer2 = nn.Linear(24, 48)
        self.layer3 = nn.Linear(48, 96)
        self.layer4 = nn.Linear(96, 192)
        self.layer5 = nn.Linear(192, 96)
        self.layer6 = nn.Linear(96, 48)
        self.layer7 = nn.Linear(48, 24)
        self.layer8 = nn.Linear(24, 12)
        self.layer9 = nn.Linear(12, 4)
        self.layer10 = nn.Linear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer9(self.activation(self.layer8(self.activation(self.layer7(self.activation(self.layer6(self.activation(self.layer5(self.activation(self.layer4(self.activation(self.layer3(self.activation(self.layer2(self.activation(self.layer1(x))))))))))))))))))
        x = self.layer10(x)
        return x



def train(model, x_train, y_train, x_test, y_test, loss_fn, optimizer, epochs=1000):
    #check out the parameters
    list(model.parameters())

    epochs = 1000

    #track different values
    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    for epoch in range(epochs):
        #set the model to training mode
        model.train() # train mode in PyTorch sets all parameters that require gradients to require gradients.

        # 1. Forward pass
        y_pred = model(x_train)

        # print(y_pred)
        # print(y_train)
        # 2. Calculate the loss
        loss = loss_fn(y_pred, y_train)
        #print(f"Loss after {epoch+1} epochs: {loss}")

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform backpropagation on the loss with respect to the parameters of the model
        loss.backward()  # calculate gradients

        # 5. Step the optimizer (perform gradient descent)
        optimizer.step() # update model parameters        by default how the optimizer changes will accumulate through the loop so.. we have to zero them above in step 3

        model.eval() # turn off gradient tracking

        with torch.inference_mode():
            test_pred = model(x_test)
            test_loss = loss_fn(test_pred, y_test)
            epoch_count.append(epoch)
            train_loss_values.append(loss)
            test_loss_values.append(test_loss)

        if epoch%10==0:
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

    print(model.state_dict())

    return epoch_count, train_loss_values, test_loss_values
