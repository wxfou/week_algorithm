import math
import numpy as np

def generateGxList(x):   #计算划分点
    gxlist = []
    for i in range(len(x) - 1):
        gx = (x[i] + x[i + 1]) / 2
        gxlist.append(gx)
    print("gxlist:"+str(gxlist))
    return gxlist

def calcAlpha(minError):                         #计算Alpha
    alpha = 1/2 * math.log((1-minError)/minError)
    print("alpha:"+str(alpha))
    return alpha

def calcNewWeight(alpha,ygx, weight, gx, y):     #计算新的权重
    newWeight = []
    sumWeight = 0
    for i in range(len(weight)):
        flag = 1
        if i  < gx and y[i] != ygx: flag = -1
        if i > gx and y[i] != -ygx: flag = -1
        weighti = weight[i]*math.exp(-alpha*flag)
        newWeight.append(weighti)
        sumWeight += weighti
    newWeight = [c/sumWeight for c in newWeight]
    print("newWeight:"+str(newWeight))
    return newWeight

def trainfxi(fx, i, x, y, weight):   #训练弱分类器
    minError = float('inf')
    bestGx = 0.5
    gxlist = generateGxList(x)
    bestygx = 1
    # 计算基本分类器
    for xi in gxlist:
        error, ygx = calcErrorNum(xi, x, y, weight)
        if error  < minError:
            minError = error
            bestGx = xi
            bestygx = ygx
    fx[i]['gx'] = bestGx
    #计算alpha
    alpha = calcAlpha(minError)
    fx[i]['alpha'] = alpha
    fx[i]['ygx'] = bestygx
    #计算新的训练数据权值
    newWeight = calcNewWeight(alpha,bestygx, weight, bestGx, y)
    return newWeight
 
def calcErrorNum(gx, x, y, weight):   #比较误差
    
    error1 = 0
    errorNeg1 = 0
    ygx = 1
    for i in range(len(x)):
        if i  < gx and y[i] != 1: error1 += weight[i]
        if i > gx and y[i] != -1: error1 += weight[i]
        if i  < gx and y[i] != -1: errorNeg1 += weight[i]
        if i > gx and y[i] != 1: errorNeg1 += weight[i]
    if errorNeg1  < error1:
        return errorNeg1, -1 
    return error1, 1 

def calcFxError(fx, n, x, y):   #计算错误率
    errorNum = 0
    for i in range(len(x)):
        fi = 0
        for j in range(n):
            fxiAlpha = fx[j]['alpha']
            fxiGx = fx[j]['gx']
            ygx = fx[j]['ygx']
            if i < fxiGx: fgx = ygx
            else: fgx = -ygx
            fi += fxiAlpha * fgx
        if np.sign(fi) != y[i]: errorNum += 1
    return errorNum/len(x)

def trainAdaBoost(x, y, errorThreshold, maxIterNum):    #训练强模型
    fx = {}
    weight = []
    xNum = len(x)
    for i in range(xNum):
        w = float(1/xNum)
        weight.append(w)
 
    for i in range(maxIterNum):
        fx[i] = {}
        newWeight = trainfxi(fx, i, x, y, weight)
        weight = newWeight
        fxError = calcFxError(fx, (i+1), x, y)
        print("fxError:"+str(fxError))
        if fxError  < errorThreshold: break
 
    return fx

def loadDataSet():     #加载数据
    x = [0, 1, 2, 3, 4, 5]
    y = [1, 1, -1, -1, 1, -1]
    return x, y
 
if __name__ == '__main__':
    x, y = loadDataSet()
    print(x)
    print(y)
    errorThreshold = 0.01    #设定错误的阈值
    maxIterNum = 10          #设置迭代次数
    fx = trainAdaBoost(x, y, errorThreshold, maxIterNum)
    print(fx)



[0, 1, 2, 3, 4, 5]
[1, 1, -1, -1, 1, -1]
gxlist:[0.5, 1.5, 2.5, 3.5, 4.5]
alpha:0.8047189562170503
newWeight:[0.1, 0.1, 0.1, 0.1, 0.5000000000000001, 0.1]
fxError:0.16666666666666666
gxlist:[0.5, 1.5, 2.5, 3.5, 4.5]
alpha:0.6931471805599453
newWeight:[0.0625, 0.0625, 0.25, 0.25, 0.31250000000000006, 0.0625]
fxError:0.16666666666666666
gxlist:[0.5, 1.5, 2.5, 3.5, 4.5]
alpha:0.7331685343967135
newWeight:[0.16666666666666666, 0.16666666666666666, 0.15384615384615385, 0.15384615384615385, 0.19230769230769235, 0.16666666666666666]
fxError:0.0
{0: {'gx': 1.5, 'ygx': 1, 'alpha': 0.8047189562170503}, 1: {'gx': 4.5, 'ygx': 1, 'alpha': 0.6931471805599453}, 2: {'gx': 3.5, 'ygx': -1, 'alpha': 0.7331685343967135}}
