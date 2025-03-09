# Understanding leNet-5 Neural Network Architecture.

# Input Layer accepts 32x32x1 grayscale images. 32 Wide, 32 tall, 1 color channel.

# Input goes through convolutional layers.

# 1st layer:
    # Convolutional Layer: 6 filters, 5x5 kernel, 1 input channel, 1 output channel.
    # Output size: 28x28x6
    # Stride: 1 pixel
    # Padding: 0 pixels (same padding)
    # Activation Function: ReLU (Rectified Linear Unit)
    # the output matrix are the dot products of each pixelxkernal.
    
    
    #Connections=(Output width)×(Output height)×(Filter size)^2 ×(Number of input maps)×(Number of filters)

    
# What that means is: 
# It has 6 layers (5x5 matrices) that slide over the image. Each filter detects a specific feature like horizontal edges, vertical edges etc.
# There are 6 layers, so it will produce 6 feature maps which are matrices.
# A stride means how many pixels at a time the filter will slide over the image. Since our stride is 1, we will go one pixel over at a time.
# It also determines the dimesions of resulting matrice. a higher stride = smaller dimensions. 5X5 with stride 2 = a 2x2. 5x5 with a stride 1 = 3x3.

# Padding adds extra pixels around the image to make sure we traverse through everything.
# After first layer, we will have 32x32x1 turned into 28x28x6, formula: (dimenstions - kernal / stride) + 1 = (32 - 5/1)+ 1 = 28. 6 is the number of kernals/filters.


#Trainable parameters are values the model learns while training.
# Main ones are Weight and Bias.

#Each of the filters have weight that respond to different parts of the image.
#for example 
    # Filter 1: Will respond to edges
    # Filter 2: Will respond to corners
    # Filter 3: Will respond to horizontal lines
    #...

# Bias is a constant value added to the output of the filter. It helps in shifting the output values around. 

# For a single filter of size 5x5, there are 25 weights. = F*F*CHANNELS = 5X5X1 = 25. We have 6 filters so total of 150 weights.
# Each filter has a bias so a total of 6 biases.

# So total parameters = Weights + Biases = 150 + 6

#Connections:
# One pixel in the output feature map is connected to the corresponding patch of 5x5. So we have 1 pixel connected to 25 pixels.
# A single feature map from the 32x32 input has 28x28 pixels.
# We have 6 feature maps from the 6 filters, so we have 28x28x6 pixels in total.
# For each pixel in [feature map], we have 26 connections (weight, 5x5 patch + bias (1)) for a total of 26 connections.
# In total, we have 28x28x6 x 26 = 122,304 connections in the first layer.



# 2nd Layer:
# Next the leNet-5 has a averaging pool layer with a filter size of 2x2 and stride of 2.
# We are essentially downsampling, reduing a 28x28 into a 14x14.
# We could use a 3x3 but that would downsample too much and rid of critial details. 2x2 is a deliberate design.

# The input of this layer is the 28x28 output from layer 1.
# Now we run a 2x2 kernal across this output. We will do averaging instead of dot product.
# We will get a 28(dimension)-2(kernal)/2(stride) + 1 = 14x14 feature map.
# We essentially downscaled it by half.
# Trainable parameters = coefficients (scaling factor: same as weights but for pooling layer) * bias * filters = 1+1*6 = 12.
# Connections = 14x14x(2x2 patch + 1 bias) = 5880 

# 3rd layer:

# Layer 3 is another/2nd convolution layer in the system.
# This one has 16 feature maps instead of 6. Also 5x5 kernal/patch and stride of 1.

# Input to this layer is the 14x14x6 output from layer 2.
# 16 filtes: n_f, F(kernal) = 5x5, stride = 1, Padding = 0, so the dimension shrinks.

# Partial connectivity: the 16 filtes do not connect to all 6 feature maps, rather each filter is connected to subset of 3-4 feature maps.
# This design reduces the number of computations and parameters while still allowing the network to learn diverse features.

# Layer 2 inputs 14x14, we have 14-5/1 + 1 = 10x10 output.
# Layer 3 also has 16 filters so it produces 16 feature maps of 10x10.
# Each filter connects to a subset, example:
    # Filter 0 connects to feature map 0,1,2
    # Filter 1 connects to feature map 1,2,3
    # Filter 2 connects to feature map 4,5,0
    # ...
    
#Partial connectivity reduces the number of weights without sacrificing much performance.
# Without partial connectivity, we would have 5x5(kernal) * 6 maps * 16 filter = 2400
# With partial connectivity, we would have  5x5(kernal) * subset of 6 maps * 16 filters= 1516.
# Instead of using all maps/filters, we use subsets.
# Next, the trainable parameters for this are : 5x5 kernal x 6 maps, x 10 filters, + the 16 bias, one for each.
# Connections = 10x10x6*10 * 10*10 = 151600
# the output is a 10x10x16 map.

# Pooling layer formula for connections: Connections=(Output width)×(Output height)×(Filter size)^2×(Number of filters)

# Fourth layer is another average pooling layer
# Same as layer 2 but instead of 6, it has 16 feature maps.
# the output will be reduced to 5x5x16
# 2x2 patch + 1 bias, so each patch contributes 5 connections.
# 5x5x5x16 = 2000 connections.

# Fifth Layer:
# Layer 5 is a convolutional layer with 120 feature maps of size 1x1. each 120 maps is connected to the previous 400 nodes 5x5x16
# Trainable parameters = (400x120) + 120 = 48120
# Each unit produces a value so we have an output of 120 units. Vector of size 120.

# Layer 6:
# Layer 6 has 84 units. it takes 120 unit vector and maps it into 84.
# each of these 84 units are connecte to output nodes. 
# weighted sum of these 84 units + a bias + a softmax activation funcntion.
#The network produces a vector with probabilities for each digit (0–9). The class with the highest probability is selected as the predicted digit.

#NOW LETS IMPLEMENT THIS.


# First lets import keras MNIST dataset for hardwritten letters.


import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset for training and testing.

# add a series of transformations using Compose.
# to tensor changes the format from numpy arrays to tensors.
# Normalize adjusts pixel value so mean is 0 and std is 128. Helps converge faster for NNs.
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,),(128,)),])

#importing data.
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)


# Load data into our batches, shuffle shuffles the data before each epoch.
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

# Now lets Define the model.

class LeNet(nn.Module):
    # Innit allows us to initialize attributes, we are creating a LetNet NN model. We can call self....functions later.
    # Super makes sure we inherit functionality from nn.Module parent class.
    def __init__(self):
        super().__init__()
        
        #1. First Convolutional Layer
        self.conv1 = nn.Conv2d(1,6,kernel_size=5, stride = 1, padding = 2)
        self.act1 = nn.Tanh()
       
       #2. Average Pooling Layer
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        #3. Second Convolutional Layer
        self.conv2 = nn.Conv2d(6,16,kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        
        #4. Average Pooling Layer
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        #5. Third Convolutional Layer
        self.convo3 = nn.Conv2d(16, 120, kernel_size=5, stride=1,padding=0)
        self.act3 = nn.Tanh()
        
        #6. Flatten and connect layers.
        self.flat = nn.Flatten() # we are changing multidimensional output into one dimensional vector.
        self.fc1 = nn.Linear(1*1*120,84) # Takes in 120 flattened inputs and outputs 84 values using linear transformation.
        self.act4 = nn.Tanh() # Adding non-linearity.
        self.fc2 = nn.Linear(84,10) # Takes in 84 inputs and outputs 10 values using linear transformation.
    
    # Forward pass through the network.
    def forward(self, x):
        
        # input 1x28x28 and output 6x28x28
        x = self.act1(self.conv1(x)) # taking the output of convolution layer 1 and putting it in a tanh function for nonlinearity.
        # input 6x28x28 and output 6x14x14
        x = self.pool1(x)
        # input 6x14x14 and output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10 and output 16x5x5
        x = self.pool2(x)
        # input 16x5x5 and output 120x1x1
        x = self.act3(self.convo3(x))
        # input 120x1x1 and output 120x1x120
        x = self.flat(x)
        # input 120x1x120 and output 84
        x = self.act4(self.fc1(x))
        # input 84 and output 10
        x = self.fc2(x)
        
        return x
    
    
model = LeNet() #creating the model.

# setting up the optimizer for training the model using the Adam optimization algorithm
# An optimizer is an algorithm used to adjust the parameters (weights and biases) of the model to minimize the loss function during training.
# We essentially run a iterator over the parameters of the model.
optimizer = optim.Adam(model.parameters())

# A loss function measures how well or poorly the model is functioning.
# it's using Cross-Entropy Loss, which is commonly used for classification problems.
loss_fn = nn.CrossEntropyLoss() 

# The model output is often in the form of logits
# CrossEntropyLoss takes these logits and applies a softmax internally to calculate probabilities

# epoch is number of iterations in the training.
n_epochs = 2

for epoch in range(n_epochs): # looping through training.
    model.train() #start training
    for x_batch, y_batch in trainloader: # loop over batches.
        y_prediction = model(x_batch) # make a prediction 
        loss = loss_fn(y_prediction, y_batch) # calculate loss between prediction and true value.
        optimizer.zero_grad() # reset gradient
        loss.backward() # calculate gradient
        optimizer.step() # update model weights.
        
    # Validation here.
    model.eval() # put in evaluation mode.
    accuracy = 0
    count = 0
    for X_batch, y_batch in testloader: # loop over batches
        y_prediction = model(X_batch) # make a prediction
        accuracy += (torch.argmax(y_prediction,1) == y_batch).float().sum() 
        # calculate accuracy using argmax which gives index of the class with highest score. AKA prediction class
        count += len(y_batch)
    accuracy = accuracy / count # calculate accuracy by dividing correct predictions / total samples.
    print("Epoch %d: model accuracy %.2f%%" %(epoch, accuracy*100))


# Here we are visualizing few samples.
model.eval()

num_samples = 5

data_iter = iter(testloader) # iterations.

images, labels = next(data_iter)

outputs = model(images)
i, predicted = torch.max(outputs, 1) # The torch.max function returns the index of the highest score in each output.

figure, grid_plots = plt.subplots(1, num_samples, figsize=(12,4))

for i in range(num_samples):      
    grid_plot = grid_plots[i]
    true_label = labels[i].item()
    prediction_label = predicted[i].item()
    grid_plot.imshow(images[i].squeeze().numpy(), cmap='gray') # Displays the i-th image in grayscale.
    grid_plot.set_title(f'True: {true_label}, \n Pred: {prediction_label}') # set title
    grid_plot.axis('off') # turn off axis.
plt.show()



