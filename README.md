# DL
#pranay
Q1
Output: Report
Code: 
Q2
#baki
SAME CODE FOR Q3 to Q8:
# pass sweep parameter in sweep_config:

# load_data() function is to load data
it normailze , mean centered and split the data for train , test and validation set
returns x_train , y_train, validation_x_test , validationj_y_test, x_test, Y_test

# arguments for NeuralNetwork object intialization
NeuralNetwork(n_input,n_output,n_hidden_layers,n_hidden_neurons)
n_input = no of input poins
n_output = no of output classes
n_hidden_layers = total no of hidden layers
n_hidden_neurons = list containing the szie of each hidden layer

## functionality of NeuralNetwork class
### fit(x_train,y_train,vx,vy,x_test,y_test,arg,optimizer,weight_ini,batch_size,epoch,lambda1,eta,run)
vx = validation x 
vy = validation y
arg = [inner activation function name ('Relu' , 'tanh' ,'sigmoid') , softmax , loss function name ('cross_entropy', 'squared_error')]
optimizer = optimizer name ('sgd' : stochastic gradient descent , 'msd' :modentum gradient descent , 'nsd': nesterov gradient descent, 'adam' , 'adagrad' ,'nadam','rmsprop')
weight_ini = initialization of weights 'random' or 'xavier'
lambda1 = regression coefficient
eta = learning rate
run = wandb.init() object i.e run = wandb.init()

### predict_and_loss(X,Y : onehot encoded ,no of points)
returns list of predicted classes , accumlated loss for given X

### accuracy_and_loss(X,Y : true label onehot encoded for X , y : true label for X)
returns accuracy , loss

### accuracy(y_o : true label for x , y_p = predicted label for x )
returns accuracy



