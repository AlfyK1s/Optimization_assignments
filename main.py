import numpy as np
from numpy .linalg import norm

MAX_ITER = 100 #max number of iterations before IPM terminates itself with "The problem does not have solution!"
alpha = 0.5

#https://github.com/AlfyK1s/Optimization_assignments - Github link

###
# Included 5 tests, which can be checked by UNCOMMENTING THE NECESSARY PARTS,
# which are marked as "UNCOMMENT THIS TO TEST" and "UNCOMMENT TILL HERE"
###

### EXAMPLE FROM LECTURE ###

# z = C^T * x
# Ax = b
# x>=0

x = np.array ( [ 2,2,4 ] , float) #initial trial solution (augmented form)
A = np.array ( [[ 1,1,1 ]] , float) # a matrix of coefficients of constraint function (augmented form)
c = np.array ( [ 1,2,0 ] , float) # vector of coefficients of objective function (augmented form)
n = 2 #num of original decision variables (before augmented form)
eps = 1e-6 #approximation accuracy
b = np.array([8])

### LPP 1
# max z = 3x1 + 2x2
# s.t.:
# -2x1 + x2 <= 2
# x1,x2 >= 0
### Converting to augmented form
# max z = 3x1 + 2x2 + 0x3
# s.t.:
# -2x1 + x2 + x3 == 2
# x1,x2,x3 >=0
### UNCOMMENT THIS TO TEST
# x = np.array([1,2,2],float)
# A = np.array ( [[ -2,1,1 ]] , float)
# c = np.array ( [ 3,2,0 ] , float)
# n = 2
# eps = 1e-6
# b = np.array([2])
### UNCOMMENT TILL HERE
# Output:
# The method is not applicable!
# Output from Simplex:
# The method is not applicable!


### LPP 2
# max z = 2x1 + 3x2
# s.t.:
# x1 + x2 <= 4
# 2x1 + x2 <= 5
# x1,x2 >= 0
### Converting to augmented form
# max z = 2x1 + 3x2 + 0x3 + 0x4
# s.t.:
# x1 + x2 + x3 == 4
# 2x1+ x2 + x4 == 5
# x1,x2,x3,x4 >=0
### UNCOMMENT THIS TO TEST
# x = np.array([1,2,1,1],float)
# A = np.array ( [[ 1,1,1,0 ],[2,1,0,1]] , float) # a matrix of coefficients of constraint function (augmented form)
# c = np.array ( [ 2,3,0,0 ] , float) # vector of coefficients of objective function (augmented form)
# n = 2 #num of original decision variables (before augmented form)
# eps = 1e-6 #approximation accuracy
# b = np.array([4,5])
### UNCOMMENT TILL HERE
# Output:
# vector of decision variables, alpha =  0.5
# 4.0149281084319457e-07
# 3.9999996572075296
# Maximum(minimum) value of the objective function: z =  11.999999774608211
#
# vector of decision variables, alpha =  0.9
# 6.038592372096618e-08
# 3.999999037899711
# Maximum(minimum) value of the objective function: z =  11.99999723447098
# Output from Simplex:
# A vector of decision variables: X∗ = [0, 4].
# Maximum value of the objective function: z = 12


### LPP 3
# max z = 4x1 + 3x2
# s.t.:
# 2x1 + x2 <= 8
# x1 + x2 <= 6
# x1,x2 >= 0
### Converting to augmented form
# max z = 4x1 + 3x2 + 0x3 + 0x4
# s.t.:
# 2x1 + x2 + x3 == 8
# x1 + x2 + x4 == 6
# x1,x2,x3,x4 >=0
### UNCOMMENT THIS TO TEST
# x = np.array([2,2,2,2],float)
# A = np.array ( [[ 2,1,1,0 ], [1,1,0,1]] , float)
# c = np.array ( [ 4,3,0,0 ] , float)
# n = 2
# eps = 1e-6
# b = np.array([8,6])
### UNCOMMENT TILL HERE
# Output:
# vector of decision variables, alpha =  0.5
# 1.999999689939381
# 3.999999973173
# Maximum(minimum) value of the objective function: z =  19.999998679276523
#
# vector of decision variables, alpha =  0.9
# 1.999999909749567
# 4.000000061051653
# Maximum(minimum) value of the objective function: z =  19.999999822153228
# Output from Simplex:
# A vector of decision variables: X∗ = [2, 4].
# Maximum value of the objective function: z = 20.


### LPP 4
# max z = 5x1 + 7x2
# s.t.:
# x1 + x2 <= 10
# 3x1 + 2x2 <= 24
# x1,x2 >= 0
### Converting to augmented form
# max z = 5x1 + 7x2 + 0x3 + 0x4
# s.t.:
# x1 + x2 + x3 == 10
# 3x1+ 2x2 + x4 == 24
# x1,x2,x3,x4 >=0
### UNCOMMENT THIS TO TEST
# x = np.array([6,2,2,2],float)
# A = np.array ( [[ 1,1,1,0 ],[3,2,0,1]] , float) # a matrix of coefficients of constraint function (augmented form)
# c = np.array ( [ 5,7,0,0 ] , float) # vector of coefficients of objective function (augmented form)
# n = 2 #num of original decision variables (before augmented form)
# eps = 1e-6 #approximation accuracy
# b = np.array([10,24])
### UNCOMMENT TILL HERE
# Output:
# vector of decision variables, alpha =  0.5
# 5.604661847242409e-07
# 9.999998718386065
# Maximum(minimum) value of the objective function: z =  69.99999383103338
#
# vector of decision variables, alpha =  0.9
# 4.383641491354301e-08
# 9.999999639626322
# Maximum(minimum) value of the objective function: z =  69.99999769656633
# Output from Simplex:
# A vector of decision variables: X∗ = [0, 10].
# Maximum value of the objective function: z = 70.


### LPP 5
# max z = x1 + x2
# s.t.:
# x1 + x2 <= 5
# x1,x2 >= 0
### Converting to augmented form
# max z = x1 + x2 + 0x3
# s.t.:
# x1 + x2 + x3 == 5
# x1,x2,x3 >=0
### UNCOMMENT THIS TO TEST
# x = np.array([2,2,1],float)
# A = np.array ( [[ 1,1,1 ]] , float) # a matrix of coefficients of constraint function (augmented form)
# c = np.array ( [ 1,1,0 ] , float) # vector of coefficients of objective function (augmented form)
# n = 2 #num of original decision variables (before augmented form)
# eps = 1e-6 #approximation accuracy
# b = np.array([5])
### UNCOMMENT TILL HERE
# Output:
# vector of decision variables, alpha =  0.5
# 2.499999761247494
# 2.4999997614924117
# Maximum(minimum) value of the objective function: z =  4.9999995227399054
#
# vector of decision variables, alpha =  0.9
# 2.499999994439264
# 2.4999999917378934
# Maximum(minimum) value of the objective function: z =  4.999999986177158
# Output from Simplex:
# A vector of decision variables: X∗ = [x1, x2], where x1 + x2 = 5 and x1, x2 ≥ 0.
# • Maximum value of the objective function: z = 5.


def solFeasibility(A,x,b):
    if np.any(np.dot(A,x)!=b):
        exit("The method is not applicable!")

def solExistence(i):
    if i==MAX_ITER:
        exit("The problem does not have solution!")

def checkUnboundness(A):
    if np.any(A<0):
        exit("The method is not applicable!")


def InteriorPointSolver(x,alpha,A,c,n,b, eps):
    i=1
    solFeasibility(A, x, b)
    checkUnboundness(A)
    while True :
        solExistence(i)
        v=x
        D = np.diag( x )

        AA = np.dot(A,D)
        cc = np.dot (D, c )
        I = np.eye ( len(c))
        F = np.dot (AA, np.transpose(AA))
        FI = np.linalg.inv(F)
        H = np.dot(np.transpose(AA),FI)
        P = np.subtract( I , np .dot(H, AA))
        cp = np.dot(P , cc)
        nu = np.absolute( np.min( cp ))
        y = np.add ( np . ones ( len(c) , float),( alpha /nu)*cp)
        yy = np.dot(D, y )
        x = yy
        i+=1
        if norm ( np.subtract(yy,v) ,ord= 2)<eps:
            break
    print("vector of decision variables, alpha = ", alpha)
    for i in range(n):
        print(x[i])
    print("Maximum(minimum) value of the objective function: z = ", c.dot(x).sum(),"\n")

InteriorPointSolver(x,alpha, A, c, n,b, eps)
InteriorPointSolver(x,alpha+0.4, A, c, n,b, eps)