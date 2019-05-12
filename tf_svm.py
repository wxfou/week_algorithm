import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()
# for iri in iris:
#     print(iris)
#     break
# for da in iris.data:
#     print (da)
x_vals=np.array([x for x in iris.data])
y_vals=np.array([1 if y==0 else -1 for y in iris.target])

train_data=np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_data=np.array(list(set(range(len(x_vals)))-set(train_data)))
x_v_train=x_vals[train_data]
x_v_test=x_vals[test_data]
y_v_train=y_vals[train_data]
y_v_test=y_vals[test_data]

batch_size=100
x_data = tf.placeholder(shape=[None,4],dtype=tf.float32)   #4列数据，行数不定
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)   #label结果

A=tf.Variable(tf.random_normal(shape=[4,1]))   #tf.random_normal用于从服从指定正太分布的数值中取出指定个数的值   
B=tf.Variable(tf.random_normal(shape=[1,1]))   #tf.Variable表示可通过对其运行操作来改变其值的张量

model=tf.subtract(tf.matmul(x_data,A),B)    
l2=tf.reduce_sum(tf.square(A))              #开方求和。即：l2正则化
alpha=tf.constant([0.01])                   #alpha变量
classification=tf.reduce_mean(tf.maximum(0.,tf.subtract(1.,tf.multiply(model,y_target))))
loss=tf.add(classification,tf.multiply(alpha,l2))

gdo=tf.train.GradientDescentOptimizer(0.01)   #通过梯度下降进行训练
to_train = gdo.minimize(loss)

saver = tf.train.Saver()

init = tf.global_variables_initializer()   #初始化变量
sess.run(init)                             #执行初始化

loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(20):
    rand_index = np.random.choice(len(x_v_train), size=batch_size)
    rand_x = x_v_train[rand_index]
    rand_y = np.transpose([y_v_train[rand_index]])
    sess.run(to_train, feed_dict={x_data: rand_x, y_target: rand_y})
    
y_test = np.reshape(y_v_test, (30,1))
array = sess.run(model,feed_dict={x_data: x_v_test, y_target: y_test})
print(array)

#采用鸢尾花数据进行预测，随机选择20%的测试集进行测试
#运行结果  [[-3.3835785][-4.97589][-7.116738][-3.5333855][-4.2969856][-7.10551][-3.4312804][-3.8654945][-3.3365154]
 #[-7.914261][-7.745061][-6.903159][-7.0140195][-7.1582117][-7.908057][-7.530942 ][-6.9372745][-7.044641][-7.433248]
 #[-5.302762][-4.750004][-5.151825][-4.9738436][-5.281886][-5.001587][-4.3969555][-4.8671165][-3.1350567][-3.683574][-4.2420197]]
