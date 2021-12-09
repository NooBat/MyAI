import numpy as np 

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# def predict(theta, X, sigma, minX) :
#     X[:, 1:] = (X[:, 1:] - minX) / sigma
#     return sigmoid(np.dot(X, theta))

# np.random.seed(10)

# theta = np.array([[0.01], [0.02], [1e-4]])
# sigma = np.array([2, 5])
# minX = np.array([31, 144])
# x0_test = np.ones((10, 1))
# x1_test = np.random.uniform(0, 8, 10).reshape((10, 1))
# x2_test = np.random.randint(70000, 100000, 10).reshape((10, 1))   

# x = np.hstack((x0_test, x1_test, x2_test)) 
# print("The inputs for testing:", x)

# predictions = predict(theta, x, sigma, minX)
# print("The output is:", predictions)

a = np.ones((10, 1))

def incre(a) : 
    a += 1            

incre(a)     
print(a)