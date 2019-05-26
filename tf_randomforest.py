from __future__ import print_function
 
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

num_steps = 500 # 总迭代次数
batch_size = 1024 # 样本batch数
num_classes = 10 # 类别数
num_features = 784 # 特征数
num_trees = 10
max_nodes = 1000
 
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int32, shape=[None])
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
 
forest_graph = tensor_forest.RandomForestGraphs(hparams)
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)
 
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

sess = tf.Session()
sess.run(init_vars)
 
# 训练
for i in range(1, num_steps + 1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
 
# 测试模型
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))



运行结果
WARNING:tensorflow:From <ipython-input-1-a09b03c32525>:9: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please use urllib or similar directly.
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting /tmp/data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
INFO:tensorflow:Constructing forest with params = 
INFO:tensorflow:{'stats_model_type': 0, 'dominate_fraction': 0.99, 'num_trees': 10, 'dominate_method': 'bootstrap', 'initialize_average_splits': False, 'split_type': 0, 'max_nodes': 1000, 'checkpoint_stats': False, 'param_file': None, 'base_random_seed': 0, 'num_output_columns': 11, 'valid_leaf_threshold': 1, 'regression': False, 'num_outputs': 1, 'num_classes': 10, 'leaf_model_type': 0, 'pruning_type': 0, 'num_splits_to_consider': 28, 'split_after_samples': 250, 'bagged_num_features': 784, 'bagged_features': None, 'feature_bagging_fraction': 1.0, 'collate_examples': False, 'finish_type': 0, 'max_fertile_nodes': 0, 'inference_tree_paths': False, 'split_pruning_name': 'none', 'bagging_fraction': 1.0, 'num_features': 784, 'use_running_stats_method': False, 'split_name': 'less_or_equal', 'model_name': 'all_dense', 'early_finish_check_every_samples': 0, 'prune_every_samples': 0, 'split_finish_name': 'basic'}
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/tensor_forest/python/ops/data_ops.py:212: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/tensor_forest/python/tensor_forest.py:606: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/contrib/tensor_forest/python/tensor_forest.py:523: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
Step 1, Loss: -1.000000, Acc: 0.431641
Step 50, Loss: -252.399994, Acc: 0.879883
Step 100, Loss: -540.799988, Acc: 0.912109
Step 150, Loss: -829.799988, Acc: 0.912109
Step 200, Loss: -1001.000000, Acc: 0.917969
Step 250, Loss: -1001.000000, Acc: 0.930664
Step 300, Loss: -1001.000000, Acc: 0.919922
Step 350, Loss: -1001.000000, Acc: 0.918945
Step 400, Loss: -1001.000000, Acc: 0.906250
Step 450, Loss: -1001.000000, Acc: 0.930664
Step 500, Loss: -1001.000000, Acc: 0.940430
Test Accuracy: 0.9189
