import json, torch, argparse
import torch.optim as optim
from nn_gen import Net
from data_gen import Data
import os

# Name: Anthony Chytros 
# ID: 20624286

# A Mulit-label Classifier using CNN in pytorch
# Modules include : nn_gen.py and data_gen.py
# Additional files include: even_mnist.csv and param.json
# Command to run code is of the format: python3 main.py files\param_file_name.json

if __name__ == '__main__':
    
    # Use argparse to pass in arguments from command line
    # Additional arguments include: data directory, verbosity
    parser = argparse.ArgumentParser(description='CNN For Multi-label Classifier')
    parser.add_argument('param', metavar='param.json', help='parameter file name')
    parser.add_argument('-data', default='even_mnist.csv', metavar='even_mnist.csv', help='data directory (default: even_mnist.csv)')
    parser.add_argument('-v', type=int, default=1, metavar='N', help='verbosity (default: 1)')
    args = parser.parse_args()
    
    # Create directory for the parameter json file
    directory = os.getcwd()
    data_directory = os.path.join(directory, args.param)
    
    # Read in json parameters 
    with open(data_directory) as paramfile:
        param = json.load(paramfile)
    
    # Initialize neural net
    model = Net()
    # Read in datafile even_mnist.csv using data_gen.py
    # Data is stored as a Data class
    data = Data(datafile = args.data, num_train = param['n_training_data'], num_test = param['n_test_data'])

    # Define an optimizer and the loss function
    # Optimizer uses Stochastic Gradient Descent
    # Loss uses Cross Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss= torch.nn.CrossEntropyLoss()
    
    # Check if GPU is available 
    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()
    
    # Initialize list of training losses, test losses and number of epochs
    train_loss= []
    test_loss= []
    num_epochs= int(param['num_epochs'])
    
    # Train the model, calculate losses, then test model on test data
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(data, loss, epoch, optimizer)
        train_loss.append(train_val)

        test_val= model.test(data, loss, epoch)
        test_loss.append(test_val)
        
        # Print more detail if verbosity is high
        if args.v>=2:
            if not ((epoch + 1) % param['display_epochs']):
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+ '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(test_val))
    
    # Print low verbosity final values 
    if args.v:
        print('Final training loss: {:.4f}'.format(train_loss[-1]))
        print('Final test loss: {:.4f}'.format(test_loss[-1]))