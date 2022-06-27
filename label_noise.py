import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sklearn.cluster
import torch.nn as nn
import torch
from torch import optim
from torch import relu as relu


###### Models ############
def HalfMSE(output, target):
    loss = (0.5) * torch.mean((output - target) ** 2)
    return loss


# 2 layer NN
class Student(nn.Module):
    """
    This is the 2-layerd neuronal network with K hidden neurons and 1 output neuron, used thoughtout this report.
    """

    def __init__(self, K, N, weight_std_initial_layer=1):
        """
        Input:  K                         = number of hidden neurons
                N                         = number of samples
                weight_std_initial_layer  = standard deviation for the weight initialization of the first
        """
        print("Creating a Student with InputDimension: %d, K: %d" % (N, K))
        super(Student, self).__init__()

        self.N = N
        self.g = nn.ReLU()
        self.K = K
        self.loss = nn.MSELoss(reduction='mean')
        # Definition of the 2 layers
        self.fc1 = nn.Linear(N, K, bias=False)
        self.fc2 = nn.Linear(K, 1, bias=False)

        ##For Figure 1 reproduction
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)

        ##For figure 4 reproduction
        # nn.init.normal_(self.fc1.weight,std=weight_std_initial_layer)
        # nn.init.normal_(self.fc2.weight,std=weight_std_initial_layer)

    def forward(self, x):
        # This is the input to the hidden layer.
        x = self.fc1(x) / math.sqrt(self.N)
        x = self.g(x)
        x = self.fc2(x)
        return x


# Oracle
def oracle(X, mu):
    """
    This function implements the 'oracle' which is defined as a network "with knowledge of the means of
    the mixture that assigns to each input the label of the nearest mean".

    Input:  X       = data points of shape [N, dim]
            mu      = means of the 4 GMs of shape [4, dim]
    Output: labels  = assigned cluster to each datapoints of shape [N]
    """
    oracle = sklearn.cluster.KMeans(n_clusters=4, init=mu, n_init=1).fit(X)
    labels = oracle.labels_
    ind1 = np.where(labels == 0)[0]
    ind2 = np.where(labels == 1)[0]
    ind3 = np.where(labels == 2)[0]
    ind4 = np.where(labels == 3)[0]

    cluster1 = np.hstack((ind1, ind2))
    cluster2 = np.hstack((ind3, ind4))

    labels[labels == 0] = -1
    labels[labels == 1] = -1
    labels[labels == 2] = 1
    labels[labels == 3] = 1

    return labels


# Random Features
def linear(x):
    return x


def centered_relu(x, var):
    a = math.sqrt(var) / math.sqrt(2 * math.pi)
    return torch.relu(x) - a


def transform_RF(X, F):
    """
    This function tansforms the datapoints X into a feature space of P>>dim, with the
    transform-matrix F.
    Input:  X       = data points of shape [N, dim]
            F       = transformation matrix of shape [dim, P]
    Output: X_trafo = transformed datapoints in the feature space of shape [N, P]
    """
    D, P = F.shape
    X = torch.from_numpy(X)
    X = X.float()
    F /= F.norm(dim=0).repeat(D, 1)
    F *= math.sqrt(D)
    X_trafo = centered_relu((X @ F) / math.sqrt(D), 0)
    return X_trafo


class Student_RF(nn.Module):
    """
    This is the second layer for the Random Features, which takes the projected datapoints
    and predcits the cluster labels via a linear model.
    """

    def __init__(self, K, N, bias=False):
        """
        Input:  K                         = number of hidden neurons
                N                         = number of samples
        """
        print("Creating a Student with InputDimension: %d, K: %d" % (N, K))
        super(Student_RF, self).__init__()

        self.P = N
        self.g = linear
        self.K = 1
        self.loss = nn.MSELoss(reduction='mean')
        self.fc1 = nn.Linear(self.P, K, bias)
        nn.init.normal_(self.fc1.weight, std=0.01)

    def forward(self, x):
        x = self.g(self.fc1(x) / math.sqrt(self.P))
        return x


########### Data Stuff ###############
def make_GMM(dim, N, var, plot, mu_r_1=None, mu_r_2=None):
    '''
    This Function generates the gaussian mixtrue models. Set plot = True to inspect the first four dimensions visually.
    input:  dim     = dimension D
            N       = number of samples
            var     = standard deviation (sigma) for all clusters
            mu_r_1  = scaling facor for the distance of the cluster centers to the origin, default=None
            mu_r_2  = scaling facor for the distance of the cluster centers to the origin, default=None
    output: X       = data points of shape [N, dim]
            Y       = labels of shape [N]
            mus     = means of the 4 GMs of shape [4, dim]
    '''
    if mu_r_1 == None:
        mu_r_1 = math.sqrt(dim)
    if mu_r_2 == None:
        mu_r_2 = math.sqrt(dim)

    # Cluster means of the 4 GMs in the first two dimensions.
    # If mu_r is set to none, then the cluster centers will be (0,±1) and (±1, 0).

    mu1 = [0, mu_r_2 / math.sqrt(dim)]
    mu2 = [0, (-1) * mu_r_2 / math.sqrt(dim)]
    mu3 = [mu_r_1 / math.sqrt(dim), 0]
    mu4 = [(-1) * mu_r_1 / math.sqrt(dim), 0]

    # Cluster means of the 4 GMs for the other D - 2 dimensions set to zero.
    if dim > 2:
        mu1 = np.append(mu1, np.zeros((dim - 2), dtype=int))
        mu2 = np.append(mu2, np.zeros((dim - 2), dtype=int))
        mu3 = np.append(mu3, np.zeros((dim - 2), dtype=int))
        mu4 = np.append(mu4, np.zeros((dim - 2), dtype=int))

    # Shared diagonal coariance matrix.

    cov = np.eye(dim) * (var ** 2)

    # Sampled datapoints from the 4 multivariate gaussians.

    cluster1 = np.random.multivariate_normal(mu1, cov, size=int(N / 4), check_valid='warn', tol=1e-8)
    cluster2 = np.random.multivariate_normal(mu2, cov, size=int(N / 4), check_valid='warn', tol=1e-8)
    cluster3 = np.random.multivariate_normal(mu3, cov, size=int(N / 4), check_valid='warn', tol=1e-8)
    cluster4 = np.random.multivariate_normal(mu4, cov, size=int(N / 4), check_valid='warn', tol=1e-8)

    # Labels for the 4 GMs according to the 2 clusters of an XOR distribution.
    label1 = np.ones(int(N / 4), dtype=int) * (-1)
    label2 = np.ones(int(N / 4), dtype=int) * (-1)
    label3 = np.ones(int(N / 4), dtype=int) * (1)
    label4 = np.ones(int(N / 4), dtype=int) * (1)

    if plot == True:
        # This part visualizes the first four dimensions of the data.

        plt.scatter(cluster1[:, 0], cluster1[:, 1], color='red')
        plt.scatter(cluster2[:, 0], cluster2[:, 1], color='red')
        plt.scatter(cluster3[:, 0], cluster3[:, 1], color='blue')
        plt.scatter(cluster4[:, 0], cluster4[:, 1], color='blue')
        plt.title('Input Space')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.gca().set_xticks([])
        plt.xticks([])
        plt.gca().set_yticks([])
        plt.yticks([])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.show()

    return np.vstack((cluster1, cluster2, cluster3, cluster4)), np.hstack((label1, label2, label3, label4)), np.vstack(
        (mu1, mu2, mu3, mu4))


def make_splits(X, Y):
    '''
    input:  X       = data points of shape [N, dim]
            Y       = labels of shape      [N]
    output: X_train = 2/3 of the datapoints used for trainig of shape     [2N/3, dim]
            X_val   = 1/3 of the datapoints used for validation of shape  [N/3, dim]
            Y_train = 2/3 of the labels used for trainig of shape         [2N/3]
            Y_val   = 1/3 of the labels used for validation of shape      [N/3]
    '''
    N = np.shape(Y)[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]
    X_train = X[0:int(N * 0.66), :]
    X_val = X[int(N * 0.66):, :]
    Y_train = Y[0:int(N * 0.66)]
    Y_val = Y[int(N * 0.66):]

    return torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(Y_train), torch.from_numpy(Y_val)


def plot_input_feature_spaces(dim, sigma, mu_r_1, mu_r_2):
    N = 500
    X, Y, m = make_GMM(dim, N, var=sigma, plot=True, mu_r_1=mu_r_1, mu_r_2=mu_r_2)
    F = torch.randn((dim, dim * 10))
    X_trafo = transform_RF(X=X, F=F)
    fig = plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_trafo[:int(N / 2), 0], X_trafo[:int(N / 2), 1], X_trafo[:int(N / 2), 2], color='red')
    ax.scatter(X_trafo[int(N / 2):int(N), 0], X_trafo[int(N / 2):int(N), 1], X_trafo[int(N / 2):int(N), 2],
               color='blue')
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')
    plt.title('Feature Space of RF')
    plt.show()


def log_sigmas(num_sigmas):
    """
  Defines the sigmas, that will be used to generate Figure 1.
  """
    sigma1 = np.logspace(-2, -1, num=int(num_sigmas / 3))
    sigma2 = np.logspace(-1, 0, num=int(num_sigmas / 3))
    sigma3 = np.logspace(0, 1, num=int(num_sigmas / 3))
    sigma = np.round(np.append(sigma1, np.append(sigma2, sigma3)), 5)
    return sigma

print("Start of script...")
N = 500
dim_NN = 100
dim_RF = 100
dim_oracle = 100
sigma = 0.01
nr_of_drawn_samples = np.round(np.linspace(0,N,num=30),0)
fraction_of_drawn_samples = np.zeros((len(nr_of_drawn_samples)))
## ORACLE:
print("Run oracle....")
oracle_pred = np.zeros((len(nr_of_drawn_samples), int(N)))
oracle_error = np.zeros((len(nr_of_drawn_samples)))
for i in range(0, len(nr_of_drawn_samples)):
    X, Y, mu = make_GMM(dim=dim_oracle, N=N, var=sigma, plot=False)
    fraction_of_drawn_samples[i] = (nr_of_drawn_samples[i]/N)*100
    draw_index = np.random.choice(X.shape[0],int(nr_of_drawn_samples[i]),replace=False)
    Y_drawn = Y[draw_index] * (-1)
    Y[draw_index] = Y_drawn
    labels = oracle(X, mu)
    ind1 = np.where(labels == 1)[0]
    ind2 = np.where(labels == -1)[0]
    oracle_error[i] = sklearn.metrics.zero_one_loss(Y, labels, normalize=True, sample_weight=None)
    oracle_pred[i, :] = labels
    print('Classifiaction error: {} for Fraction of noise: {}'.format(np.round(oracle_error[i], 3),fraction_of_drawn_samples[i]))
print("Oracle calcluations are finished!...")
#label = [-1,1]
#colors = ['red','blue']
#fig = plt.figure(figsize=(4,4))
#plt.scatter(X[:,0], X[:,1], c=Y, cmap=matplotlib.colors.ListedColormap(colors))
## RANDOM FEATURES
print("Start Random Features Training...")
P = 2 * dim_RF  # projection dimension
reg_RF = 0.0  # regulaization parameter
lr = 0.5  # learning rate
RF_error = np.zeros((len(nr_of_drawn_samples)))

######### initilize the second layer  for RF #########################
######################################################################
student = Student_RF(N=P, K=1)
params = []
params += [{'params': student.fc1.parameters(), 'lr': lr, 'weight_decay': reg_RF}]
optimizer = optim.SGD(params, lr=lr, weight_decay=reg_RF)
criterion = student.loss

######### iterate over the sigmas  ###################################
######################################################################

for i in range(0, len(nr_of_drawn_samples)):
    X_, Y_, mu = make_GMM(dim=dim_RF, N=N, var=sigma, plot=False)
    fraction_of_drawn_samples[i] = (nr_of_drawn_samples[i] / N) * 100
    draw_index = np.random.choice(X.shape[0], int(nr_of_drawn_samples[i]), replace=False)
    Y_drawn = Y[draw_index] * (-1)
    Y[draw_index] = Y_drawn
    F = torch.randn((dim_RF, P))  # random, but fixed projection matrix
    X = transform_RF(X_, F)  # projected data
    X = (X).numpy()

    X_train, X_val, Y_train, Y_val = make_splits(X, Y)
    X_val = (X_val).float()
    Y_val = (Y_val).float()

    ######### Training the RF with online SGD on halfMSE #################
    ######################################################################
    student.train()
    for j in range(X_train.shape[0]):
        targets = (Y_train[j]).float()
        inputs = (X_train[j, :]).float()
        student.zero_grad()
        preds = student(inputs)
        loss = HalfMSE(preds, targets)
        if j% 500 ==0: #print train loss every 100 steps
          print("Train loss: {}".format(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        optimizer.step()

    ######### Evaluation of the training of RF on the classification error
    ######################################################################
    student.eval()
    with torch.no_grad():
        preds = student(X_val)
        preds = preds[:, 0]
        eg = HalfMSE(preds, Y_val)
        # calculate the classification error with the predictions
        eg_class = 1 - torch.relu(torch.sign(preds) * Y_val)
        eg_class = eg_class.sum() / float(preds.shape[0])
        # print("preds:{}, y_val:{}".format(preds,Y_val))
        RF_error[i] = eg_class
        print("Test Data: Classification Error: {}; Fraction of noise: {}; halfMSE-Loss:{}".format(np.round(RF_error[i], 3),
                                                                                          fraction_of_drawn_samples[i], eg))
        print("---------------------------------------------------------")

print("Random Features Training is finished!..")

## NEURAL NETWORK
print("Start Neural Network Training...")
K = 12  # number of hidden neurons
lr_NN = 0.1  # learning rate
reg_NN = 0.0  # regulaization parameter
NN_error = np.zeros((len(nr_of_drawn_samples)))

######### initilize the 2LNN #########################################
######################################################################
student = Student(K=K, N=dim_NN)
params = []
params += [{'params': student.fc1.parameters()}]
params += [{'params': student.fc2.parameters(), 'lr': lr_NN, 'weight_decay': reg_NN}]
optimizer = optim.SGD(params, lr=lr_NN, weight_decay=reg_NN)  # Define which parameters should be optimized by the SGD
criterion = student.loss

######### iterate over the sigmas  ###################################
######################################################################
for i in range(0, len(nr_of_drawn_samples)):
    X, Y, m = make_GMM(dim=dim_NN, N=N, var=sigma, plot=False)
    fraction_of_drawn_samples[i] = (nr_of_drawn_samples[i] / N) * 100
    draw_index = np.random.choice(X.shape[0], int(nr_of_drawn_samples[i]), replace=False)
    Y_drawn = Y[draw_index] * (-1)
    Y[draw_index] = Y_drawn
    X_train, X_val, Y_train, Y_val = make_splits(X, Y)
    X_train = X_train
    Y_train = Y_train
    X_val = (X_val).float()
    Y_val = (Y_val).float()

    ######### Training the NN with online SGD on halfMSE #################
    ######################################################################
    student.train()
    for j in range(X_train.shape[0]):
        targets = (Y_train[j]).float()
        inputs = (X_train[j, :]).float()
        student.zero_grad()
        preds = student(inputs)
        loss = HalfMSE(preds, targets)
        if j% 500 ==0: #print train loss every 100 steps
          print("Train loss: {}".format(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()

    ######### Evaluation of the training of NN on the classification error
    ######################################################################
    student.eval()
    with torch.no_grad():
        preds = student(X_val)
        preds = preds[:, 0]
        eg = HalfMSE(preds, Y_val)
        # calculate the classification error with the predictions
        eg_class = 1 - torch.relu(torch.sign(preds) * Y_val)
        eg_class = eg_class.sum() / float(preds.shape[0])
        NN_error[i] = eg_class
        print("Test Data: Generalized Classification Error: {}; Fraction of noise: {}; Loss:{}".format(np.round(NN_error[i], 3),
                                                                                              fraction_of_drawn_samples[i],
                                                                                              eg))
        print("---------------------------------------------------------")
print("NN Training is finished!")
print("Save all results")
with open('/home/apdl007/Paper2_extension/oracle_error.txt', 'w') as f:
    np.savetxt(f, oracle_error)
with open('/home/apdl007/Paper2_extension/2LNN_error.txt', 'w') as f:
    np.savetxt(f, NN_error)
with open('/home/apdl007/Paper2_extension/RF_error.txt', 'w') as f:
    np.savetxt(f, RF_error)
with open('/home/apdl007/Paper2_extension/fraction.txt', 'w') as f:
        np.savetxt(f, fraction_of_drawn_samples)