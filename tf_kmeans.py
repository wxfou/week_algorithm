import tensorflow as tf
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles

K=4
max_iter=1000
N=100
centers = [[-2, -2], [-2, 1.5], [1.5, -2], [2, 1.5]]  # 簇中心
data, features = make_blobs(n_samples=N, centers=centers, n_features = 2, cluster_std=0.8, shuffle=False, random_state=42) #产生聚类算法的数据  

def class_mean(data,id,num):    #求组内均值
    total=tf.unsorted_segment_sum(data,id,num)
    count=tf.unsorted_segment_sum(tf.ones_like(data),id,num)
    return total/count

points=tf.Variable(data)
cluster=tf.Variable(tf.zeros([N],dtype=tf.int64))
centers=tf.Variable(tf.slice(points.initialized_value(),[0,0],[K,2]))  # 初始化前k个点当做初始中心
repcenters=tf.reshape(tf.tile(centers,[N,1]),[N,K,2])    #复制中心
reppoints=tf.reshape(tf.tile(points,[1,K]),[N,K,2])      #复制聚类数据
sumSqure=tf.reduce_sum(tf.square(repcenters-reppoints),reduction_indices=2)  #计算所以的距离之和
bestCenter=tf.arg_min(sumSqure,1)    # 寻找最近的簇中心
change=tf.reduce_any(tf.not_equal(bestCenter,cluster))  #计算两个类别中心的变化情况
means=class_mean(points,bestCenter,K)   #求组内均值
with tf.control_dependencies([change]):  # 将组内均值变成新的簇中心，同时分类结果也要更新
    update=tf.group(centers.assign(means),cluster.assign(bestCenter))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   #初始化所以值
    changed=True
    iterNum=0
    while changed and iterNum<max_iter:  #进行迭代更新
        iterNum+=1
        [changed, _] = sess.run([change, update])
        [centersArr, clusterArr] = sess.run([centers, cluster])
        print(clusterArr)
        print(centersArr)
        
        
#运行结果如下：
<tf.Variable 'Variable_51:0' shape=(100, 2) dtype=float64_ref>
<tf.Variable 'Variable_52:0' shape=(100,) dtype=int64_ref>
<tf.Variable 'Variable_53:0' shape=(4, 2) dtype=float64_ref>
[0 1 2 3 2 2 2 2 2 2 0 2 2 2 2 1 2 0 2 2 0 2 2 1 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 1 1 3 3 3 3 3 3 3 3 1 3 1 3 3 3 3 3 3 3 3]
[[-1.38128117 -2.50824898]
 [-1.65512972  1.47829377]
 [-2.47631665 -2.44011099]
 [ 1.65418048 -0.38299136]]
[0 0 2 0 2 2 0 2 2 2 0 0 2 2 2 2 2 0 0 2 0 2 2 2 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 3 3 3 3 0 3 3 3 3 3 3 3 0 3 0 3 3 3 3 3 0 3 0
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
[[-0.79287638 -2.41322534]
 [-2.07324791  1.60169731]
 [-2.59237506 -2.03173978]
 [ 1.85585694  0.15379672]]
[0 2 2 0 2 2 0 2 2 2 0 2 2 2 2 2 2 0 0 2 0 2 2 2 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 3 3 3 0 3 3 0 3 3 3 3 0 0 0 0 3 0 0 3 0 3 0
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
[[-0.06530236 -2.47666211]
 [-2.07324791  1.60169731]
 [-2.48902662 -2.02338067]
 [ 1.93961993  0.56868597]]
[2 2 2 0 2 2 2 2 2 2 0 2 2 2 2 2 2 0 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 3 0 0 3 0 0 3 0 0 3 0 0 0 0 3 0 0 3 0 3 0
 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
[[ 0.86504715 -2.2932102 ]
 [-2.07324791  1.60169731]
 [-2.30407966 -2.22180821]
 [ 1.95655687  1.08489213]]
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 3 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0
 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
[[ 1.3325916  -2.08529293]
 [-2.07324791  1.60169731]
 [-2.1438331  -2.21692515]
 [ 1.96415845  1.52089702]]
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 3 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0
 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
[[ 1.3325916  -2.08529293]
 [-2.07324791  1.60169731]
 [-2.1438331  -2.21692515]
 [ 1.96415845  1.52089702]]
