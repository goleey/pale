import random
import sys
from time import time

import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
# method = "line1"
# method = "deepwalk"
# embedding1 = object_op.load_object("../object/aminer_"+ method+"_emb.pkl").T.to_dict("list")
# embedding2 = object_op.load_object("../object/linkedin_" + method + "_emb.pkl").T.to_dict("list")
embedding1_file, embedding2_file, linear, training_file, validation_file, test_file, mapping_source_embedding_out= sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7]
def read_embedding(filename):
    with open(filename) as f:
        dic = {}
        lines = f.readlines()
        for line in lines[1:]:
            sp = re.split("\\s+", line.strip())
            dic[sp[0]] = np.array([float(x) for x in sp[1:]])
    return dic
def read_dataset(dataset_file):
    dataset = {}
    with open(dataset_file) as f:
        lines = f.readlines()
        for line in lines:
            sp = re.split("\\s+", line.strip())
            dataset[sp[0]] = sp[1]
    return dataset
def read_test(test_file):
    test = {}
    with open(test_file) as f:
        lines = f.readlines()
        for line in lines:
            sp = re.split("\\s+", line.strip())
            test[sp[0]] = sp[1:]
    return test
# filter those node may not have embedding
def filter_data(dataset_file, embedding1, embedding2):
    dataset = read_dataset(dataset_file)
    data_new = []
    for item in dataset.items():
        if item[0] in embedding1.keys() and item[1] in embedding2.keys():
            data_new.append(item)
        if item[0] not in embedding1.keys():
            print "network1:", item[0]
        if item[1] not in embedding2.keys():
            print "network2:", item[1]
    return dict(data_new)

def get_batches(training_net_set, batch_size, embedding1, embedding2):
    training_net_set = training_net_set.items()
    random.shuffle(training_net_set)
    train_length = len(training_net_set)
    # fold = train_length / batch_size
    batches_x = []
    batches_y = []
    index = 0
    while index < train_length:
        # end = index
        if index + batch_size >= train_length:
            end = train_length
        else:
            end =  index + batch_size + 1
        batches = training_net_set[index: end]
        batch_x, batch_y = get_batch_x_y(batches, embedding1, embedding2)
        batches_x.append(batch_x)
        batches_y.append(batch_y)
        index = end
    return batches_x, batches_y, len(batches_x)

def get_batch_x_y(batches, embedding1, embedding2):
    batch_x = []
    batch_y = []
    for item in batches:
        batch_x.append([float(x) for x in embedding1[item[0]]])
        batch_y.append([float(x) for x in embedding2[item[1]]])

    return batch_x, batch_y


def get_validation(validation_net_set, embedding1, embedding2):
    batch_x = []
    batch_y = []
    for item in validation_net_set.items():
        batch_x.append([float(x) for x in embedding1[item[0]]])
        batch_y.append([float(x) for x in embedding2[item[1]]])
    return batch_x, batch_y

def filter_test(test_file, embedding1, embedding2):
    dataset = read_test(test_file)
    data_new = []
    for item in dataset.items():
        key = item[0]
        groundtruth = item[1][0]
        if key in embedding1 and groundtruth in embedding2:
            data_new.append(item)
    return dict(data_new)
# get the groundtruth
def get_test(test_net_set, embedding1, embedding2):
    batch_x = []
    batch_y = []
    for item in test_net_set.items():
        # print item
        batch_x.append([float(x) for x in embedding1[item[0]]])
        batch_y.append([float(x) for x in embedding2[item[1][0]]])
    return batch_x, batch_y

def get_test_data(testing_set, embedding1, embedding2):
    source_node_id = []
    test_x = []
    test_y_list = []
    for item in testing_set.items():
        # if item[0] in embedding1:
        source_node_id.append(item[0])
        test_x.append([float(x) for x in embedding1[item[0]]])
        test_y = []
        for id in item[1]:
            if id in embedding2:
                embedding = embedding2[id]
            else:
                embedding = [0] * num_classes
                print "missing_id---" + id
            test_y.append([float(x) for x in embedding])
        test_y_list.append(test_y)
    return test_x, test_y_list, source_node_id

def save_embedding(dic, filename):
    with open(filename, "w+") as f:
        for k,v in dic.items():
            f.write(k + " ")
            for item in v:
                f.write(str(item) + " ")
            f.write("\n")

embedding1 = read_embedding(embedding1_file)
embedding2 = read_embedding(embedding2_file)

learning_rate = 0.001
batch_size = 512
epochs = 3000
display_step = 100
n_hidden_1 = len(embedding1.items()[0][1]) * 2
num_input = len(embedding1.items()[0][1])
num_classes = len(embedding2.items()[0][1])
print num_input, n_hidden_1, num_classes

# train a mapping function and validate it, and then test the performance.
def train_vali():

    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    weights = {
        "h1": tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        "out": tf.Variable(tf.random_normal([n_hidden_1, num_classes])),

        "out_one": tf.Variable(tf.random_normal([num_input, num_classes]))
    }
    biases = {
        "b1": tf.Variable(tf.random_normal([n_hidden_1])),
        "out": tf.Variable(tf.random_normal([num_classes]))
    }

    def linear_neural_net(x):
        out_layer = tf.matmul(x, weights["out_one"])
        return out_layer

    def non_linear_neural_net(x):
        layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights["h1"]), biases["b1"]))
        out_layer = tf.sigmoid(tf.matmul(layer_1, weights["out"]) + biases["out"])
        return out_layer

    if linear == "1":
        print "linear function"
        logits = linear_neural_net(X)
        loss_op = tf.nn.l2_loss(logits - Y)
    else:
        print "non linear function"
        logits = non_linear_neural_net(X)
        loss_op = tf.nn.l2_loss(logits - Y) + layers.l2_regularizer(0.01)(weights["out_one"])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    print "network constructed done!"

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        training_net_set = filter_data(training_file, embedding1, embedding2)
        validation_net_set = filter_data(validation_file, embedding1, embedding2)

        train_len = len(training_net_set)
        validation_len = len(validation_net_set)

        print "train size:", train_len
        print "vali size:", validation_len

        # batches_x, batches_y, batch_len = get_batches(training_net_set, batch_size, embedding1, embedding2)
        train_x, train_y = get_validation(training_net_set, embedding1, embedding2)
        validation_x, validation_y = get_validation(validation_net_set, embedding1, embedding2)
        for epoch in range(epochs):
            # batch_x, batch_y = next_batch(training_net_set, batch_size, embedding1, embedding2)
            batches_x, batches_y, batch_len = get_batches(training_net_set, batch_size, embedding1, embedding2)
            for step in range(len(batches_x)):
                batch_x = batches_x[step]
                batch_y = batches_y[step]
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step == 0 or step == len(batches_x) - 1:
                    train_loss = sess.run(loss_op, feed_dict={X: train_x, Y: train_y})
                    vali_loss = sess.run(loss_op, feed_dict={X: validation_x, Y: validation_y})
            if epoch%100 == 0:
                print ("epoch " +str(epoch) + ", train loss= " + "{:.4f}".format(train_loss/train_len) + ", vali loss= " + "{:.4f}".format(vali_loss/validation_len) +", gap= " + "{:.4f}".format(train_loss/train_len - vali_loss/validation_len))

        print "validation loss: " + str(sess.run(loss_op, feed_dict={X: validation_x, Y: validation_y}) / validation_len)
        testing_net_set = filter_test(test_file, embedding1, embedding2)
        test_x, test_y = get_test(testing_net_set, embedding1, embedding2)
        test_len = len(test_y)
        print "test loss: " + str(sess.run(loss_op, feed_dict={X: test_x, Y: test_y}) / test_len)

        test_x, test_y_list, source_node_id = get_test_data(testing_net_set, embedding1, embedding2)
        print "test_size:", test_len

        predicted_embedding = sess.run(logits, feed_dict = {X: test_x}).tolist()
        predict_dic = {}
        for i in range(len(test_x)):
            predict_dic[source_node_id[i]] = predicted_embedding[i]
        print "save test set embedding..."
        save_embedding(predict_dic, mapping_source_embedding_out)

        # calculate the mrr and hit@1 of testset
        dis_list = []
        y_pred = []
        # for item in test_y_list:
        #     dis_list.append()
        for i in range(len(test_y_list)):
            # embedding_s = np.array(test_x[i])
            candidates = test_y_list[i]
            embedding_s_mapping = np.array(predicted_embedding[i])
            dis_one = []
            for candidate in candidates:
                candidate = np.array(candidate)
                dis_one.append(np.linalg.norm(embedding_s_mapping - candidate))
            dis_list.append(dis_one)

        mrr = 0.0
        for dis_one in dis_list:
            dis_items = []
            i = 0
            for item in dis_one:
                dis_items.append((i, item))
                i += 1
            sort_dis = sorted(dis_items, key=lambda x: x[1], reverse=False)
            min_index = sort_dis[0][0]
            mrr += 1.0 / (min_index + 1)
            if min_index == 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        print "mrr:", mrr/len(test_x)
        print "hit@1:", accuracy_score(np.ones(len(y_pred)), y_pred)
        sess.close()


if __name__ == '__main__':
    t1 = time()
    train_vali()
    t2 = time()
    print "time consuming:", t2 - t1
