from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np

from utils import *
from models import GCN, MLP


class flag_class():
    dataset = 'cora'        # Dataset string :'cora', 'citeseer', 'pubmed'
    model = 'gcn'           # Model string. :'gcn', 'gcn_cheby', 'dense'
    learning_rate = 0.01    # Initial learning rate
    epochs = 200            # Number of epochs to train
    hidden1 = 16            # Number of units in hidden layer 1
    dropout = 0.5           # Dropout rate (1 - keep probability)
    weight_decay = 5e-4     # Weight for L2 loss on embedding matrix
    early_stopping = 10     # Tolerance for early stopping (# of epochs)
    max_degree = 3          # Maximum Chebyshev polynomial degree
    checkpoint_every = 20   # Save model after this many steps
    num_checkpoints = 5     # Number of checkpoints to store
    save_best = True        # Save the best model



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
FLAGS = flag_class()


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
model = model_func(
    placeholders, input_dim=features[2][1], FLAGS=FLAGS, logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
best_acc = 0
steps = 0
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy],
                    feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    steps += 1

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    if steps % FLAGS.checkpoint_every == 0:
        if acc > best_acc:
            best_acc = acc
            if FLAGS.save_best:
                model.save(sess=sess, save_dir='snapshot',
                           save_tittle='best', steps=steps)
        else:
            model.save(sess=sess, save_dir='snapshot',
                       save_tittle='snapshot', steps=steps)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
