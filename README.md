# DL
## Final code file is named as # final_dl_assignment1
'''3k & sonagara'''

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

### forward_pass(x)
to forward pass 

### gradient(x,y)
it calls forward pass and do the backward pass

### onehot_encoding( y : true label for x, no of output classes)
returns onehot array for y

### add(d1 : dictionay , d2 : dictionay , m1 : mutiply to d1 , m2 : multiply to d2) 
returns dictionary

### mul(d1: dictionay , m1 : a no to multiply to d1)
returns dictionay

### squr(d1: dictinary)
returns dictionar

### adarate (d1:dict , d2 : dict , m1 : no , m2 : no ) 
returns m1*d1/(m2*d2)

# adding new optimizers
use oW and ob to initialize weights and bias to zero for some new vaiable ex: dw = self.oW
in some cases this might not work for that we can use the below code (when the variables are local):
 v_w , v_b = {},{}
 for i in range(self.no_hidden_layers+1):
   v_w[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
   v_b[i+1] = np.zeros((1, self.sizes[i+1]))
 
## important : 
once the weight update is done 
copy this weights to self.tW with required modification(as the gradient are calculated for tW)
this is helpful for the optimizers which calculates the derivative at look up like in nesterov gradient descent
so after doing W -= dw
do self.tW = W + look ahed
(see the availabel optimizers for better understanding)

# to train the model
call train()

   
   
  
  




