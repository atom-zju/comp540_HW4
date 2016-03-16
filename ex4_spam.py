from sklearn import preprocessing, metrics
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass
from sklearn.cross_validation import train_test_split


#############################################################################
# load the SPAM email training and test dataset                             #
#############################################################################

X,y = utils.load_mat('data/spamTrain.mat')
yy = np.ones(y.shape)
yy[y==0] = -1

test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest'].flatten()

#############################################################################
# your code for setting up the best SVM classifier for this dataset         #
# Design the training parameters for the SVM.                               #
# What should the learning_rate be? What should C be?                       #
# What should num_iters be? Should X be scaled? Should X be kernelized?     #
#############################################################################
# your experiments below

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)

print X_train.shape
print X_val.shape

svm = LinearSVM_twoclass()
svm.theta = np.zeros((X.shape[1],))

'''
    first select the best gaussian kernelized svm
'''

Cvals = [0.01,0.03,0.1,0.3,10,30]
sigma_vals = [0.01,0.03,0.1,0.3,10,30]
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1]
iterations = [100 ,1000, 10000]

best_acc = 0
for sigma_val in sigma_vals:
    K = np.array([utils.gaussian_kernel(x1,x2,sigma_val) for x1 in X_train for x2 in X_train]).reshape(X_train.shape[0],X_train.shape[0])
    scaler = preprocessing.StandardScaler().fit(K)
    scaleK = scaler.transform(K)
    KK = np.hstack([np.ones((scaleK.shape[0],1)),scaleK])
    Kval = np.array([utils.gaussian_kernel(x1,x2,sigma_val) for x1 in X_val for x2 in X_train]).reshape(X_val.shape[0],X_train.shape[0])
    scaleKval = scaler.transform(Kval)
    KKval = np.hstack([np.ones((scaleKval.shape[0],1)),scaleKval])
    for Cval in Cvals:
        for learning_rate in learning_rates:
            for iteration in iterations:
                svm = LinearSVM_twoclass()
                svm.theta = np.zeros((KK.shape[1],))
                svm.train(KK,y_train,learning_rate=learning_rate,C=Cval,num_iters=iteration,verbose=True)
                y_pred = svm.predict(KKval)
                acc = np.mean(y_pred == y_val)
                if acc > best_acc:
                    best_acc = acc
                    best_C=Cval
                    best_sigma = sigma_val
                    best_iteration = iteration
                    best_learning_rate = learning_rate

print "For gussian kernel: "
print "the best acc is ", best_acc
print "the best c is ",  best_C
print "the best sigma is ", best_sigma
print "the best learning rate is ", best_learning_rate
print "the best iteration is ", best_iteration



'''
    first select the best linear svm
'''

Cvals = [0.01,0.03,0.1,0.3,10,30]
sigma_vals = [0.01,0.03,0.1,0.3,10,30]
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1]
iterations = [100 ,1000, 10000]

best_acc = 0
for Cval in Cvals:
    for learning_rate in learning_rates:
        for iteration in iterations:
            svm = LinearSVM_twoclass()
            svm.theta = np.zeros((X_train.shape[1],))
            svm.train(X_train,y_train,learning_rate=learning_rate,C=Cval,num_iters=iteration,verbose=True)
            y_pred = svm.predict(X_val)
            acc = np.mean(y_pred == y_val)
            if acc > best_acc:
                best_acc = acc
                best_C=Cval
                best_iteration = iteration
                best_learning_rate = learning_rate

print "For linear kernel: "
print "the best acc is ", best_acc
print "the best c is ",  best_C
print "the best learning rate is ", best_learning_rate
print "the best iteration is ", best_iteration



'''
    first select the best qaurdratic kernelized svm
'''

Cvals = [0.01,0.03,0.1,0.3,10,30]
c_s = [0, 0.1,1,10]
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1]
iterations = [100, 1000, 10000]

best_acc = 0
for c_ in c_s:
    K = np.array([utils.quadraric_kernel(x1,x2,c_) for x1 in X_train for x2 in X_train]).reshape(X_train.shape[0],X_train.shape[0])
    scaler = preprocessing.StandardScaler().fit(K)
    scaleK = scaler.transform(K)
    KK = np.hstack([np.ones((scaleK.shape[0],1)),scaleK])
    Kval = np.array([utils.quadraric_kernel(x1,x2,c_) for x1 in X_val for x2 in X_train]).reshape(X_val.shape[0],X_train.shape[0])
    scaleKval = scaler.transform(Kval)
    KKval = np.hstack([np.ones((scaleKval.shape[0],1)),scaleKval])
    for Cval in Cvals:
        for learning_rate in learning_rates:
            for iteration in iterations:
                svm = LinearSVM_twoclass()
                svm.theta = np.zeros((KK.shape[1],))
                svm.train(KK,y_train,learning_rate=learning_rate,C=Cval,num_iters=iteration,verbose=True)
                y_pred = svm.predict(KKval)
                acc = np.mean(y_pred == y_val)
                if acc > best_acc:
                    best_acc = acc
                    best_C=Cval
                    best_c_ = c_
                    best_iteration = iteration
                    best_learning_rate = learning_rate

print "For qaudraic kernel: "
print "the best acc is ", best_acc
print "the best c is ",  best_C
print "the best c_ is ", c_
print "the best learning rate is ", best_learning_rate
print "the best iteration is ", best_iteration

#############################################################################
#  end of your code                                                         #
#############################################################################


#############################################################################
# what is the accuracy of the best model on the training data itself?       #
#############################################################################
# 2 lines of code expected

y_pred = svm.predict(X)
print "Accuracy of model on training data is: ", metrics.accuracy_score(yy,y_pred)


#############################################################################
# what is the accuracy of the best model on the test data?                  #
#############################################################################
# 2 lines of code expected


yy_test = np.ones(y_test.shape)
yy_test[y_test==0] = -1
test_pred = svm.predict(X_test)
print "Accuracy of model on test data is: ", metrics.accuracy_score(yy_test,test_pred)


#############################################################################
# Interpreting the coefficients of an SVM                                   #
# which words are the top predictors of spam?                               #
#############################################################################
# 4 lines of code expected

words, inv_words = utils.get_vocab_dict()

index = np.argsort(svm.theta)[-15:]
print "Top 15 predictors of spam are: "
for i in range(-1,-16,-1):
    print words[index[i]+1]


