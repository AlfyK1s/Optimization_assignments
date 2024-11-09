import numpy as np
import copy
supplies = np.array([20, 15, 25],int)
costs = np.array([[6,4,1,5],[3,8,7,4],[5,2,3,9]],int)
demand = np.array([10,15,20,15],int)
suppliesUsed = np.zeros((3,4),int)
TotalSum = np.sum(demand)
def checkBalanced(supplies,demand):
    if np.sum(supplies)!=np.sum(demand):
        exit("The problem is not balanced!")

checkBalanced(supplies,demand)
def getColSum(col):
    sum=0
    for a in suppliesUsed:
        sum+=a[col]
    return sum
def NorthWestMethod(supply,cost,demand_list,suppliesUsed_list,nRows,nCols):
    supplies = np.copy(supply)
    costs = np.copy(cost)
    demand = np.copy(demand_list)
    suppliesUsed = np.copy(suppliesUsed_list)
    row=0
    col=0
    while row != len(costs) and col !=len(costs[0]):
        if supplies[row] <=demand[col]:
            suppliesUsed[row][col] = supplies[row]
            demand[col] -= supplies[row]
            row+=1
        else:
            suppliesUsed[row][col] = demand[col]
            supplies[row] -=demand[col]
            col+=1
    print("North-West Method")
    print(suppliesUsed)

def findRowDiff(cost):
    costs = np.copy(cost)
    rowD=np.array([0,0,0],int)
    copyCosts= np.copy(costs)
    for i in range(3):
        copyCosts[i].sort()

        rowD[i]=(copyCosts[i][1]-copyCosts[i][0])
    return rowD
def findColDiff(cost):
    costs = np.copy(cost)
    rowD=np.array([0,0,0,0],int)
    copyCosts= np.copy(costs).transpose()
    for i in range(4):
        copyCosts[i].sort()

        rowD[i]=(copyCosts[i][1]-copyCosts[i][0])
    return rowD
def findMaxDiff(cost):
    costs = np.copy(cost)
    rowD = findRowDiff(costs).tolist()
    colD = findColDiff(costs).tolist()
    return rowD,colD
def VogelMethod(supply,cost,demand_list,suppliesUsed_list,nRows, nCols):
    supplies = np.copy(supply)
    costs = np.copy(cost)
    demand = np.copy(demand_list)
    suppliesUsed = np.copy(suppliesUsed_list)
    while supplies.max()!=0 or demand.max()!=0:
        rowD,colD = findMaxDiff(costs)
        if max(rowD) >= max(colD):
            for i, v in enumerate(rowD):
                if v == max(rowD):
                    min1 = min(costs[i])
                    for ind, val in enumerate(costs[i]):
                        if val == min1:
                            min2 = min(supplies[i],demand[ind])
                            suppliesUsed[i][ind]+= min2
                            supplies[i] -= min2
                            demand[ind] -= min2
                            if(demand[ind] == 0 ):
                                for k in range(nRows):
                                    costs[k][ind]  =  1000
                            else:
                                costs[i] = [100 for i in range(nCols)]
                            break
        else:
            for ind, val in enumerate(colD):
                if val ==max(colD):
                    min1= 1000
                    for j in range(nRows):
                        min1 =  min(min1, costs[j][ind])
                    for ind2 in range(nRows):
                        val2 = costs[ind2][ind]
                        if val2 == min1:
                            min2 = min(supplies[ind2],demand[ind])
                            suppliesUsed[ind2][ind]+= min2
                            supplies[ind2] -=min2
                            demand[ind] -= min2
                            if demand[ind] ==0:
                                for k in range(nRows):
                                    costs[k][ind]  =  1000
                            else:
                                costs[ind2] = [1000 for i in range (nCols)]
                            break
                    break
    print("Vogel's method")
    print(suppliesUsed)

def RusselsMethod(supply, cost, demand_copy,suppliesUsed_copy):
    supplies = np.copy(supply)
    costs = np.copy(cost)
    demand = np.copy(demand_copy)
    suppliesUsed = np.copy(suppliesUsed_copy)
    while np.any(supplies) and np.any(demand):
        getUarr(costs)
        getVarr(costs)
        u_array = np.array(getUarr(costs)) # rows - 3
        v_array = np.array(getVarr(costs)) # cols- 4
        delta_array= np.zeros((3,4),int)
        for a in range(3):
            for b in range(4):
                delta_array[a][b] = costs[a][b] - u_array[a] - v_array[b]
        min_delta= delta_array.min()
        min_delta_r,min_delta_c = np.where(delta_array==min_delta)
        min_row = int(min_delta_r[0])
        min_col =  int( min_delta_c[0])
        if supplies[min_row]<=demand[min_row]:
            suppliesUsed[min_row][min_col] += supplies[min_row]
            costs[min_row][min_col] += 10000
            demand[min_col] -=supplies[min_row]
            supplies[min_row] = 0
        else:
            suppliesUsed[min_row][min_col] += demand[min_col]
            costs[min_row][min_col] += 10000
            supplies[min_row]-= demand[min_col]
            demand[min_col] = 0
    print("Russels method")

    print(suppliesUsed)
def getUarr(cost):
    costs = np.copy(cost)
    u_array=[0,0,0]
    for i in range(3):
        u_array[i] = max(costs[i])
    return u_array
def getVarr(cost):
    costs = np.copy(cost).transpose()
    v_array=[0,0,0,0]

    for i in range(4):

        v_array[i] = max(costs[i])
    return v_array
VogelMethod(supplies, costs, demand, suppliesUsed, 3, 4)
NorthWestMethod(supplies,costs,demand,suppliesUsed,3,4)
RusselsMethod(supplies,costs,demand,suppliesUsed)
