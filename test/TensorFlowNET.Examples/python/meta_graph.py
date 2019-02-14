
import tensorflow as tf
import math

# Creates an inference graph.
# Hidden 1
images = tf.constant(1.2, tf.float32, shape=[100, 28])
with tf.name_scope("hidden1"):
  weights = tf.Variable(
      tf.truncated_normal([28, 128],
                          stddev=1.0 / math.sqrt(float(28))),
      name="weights")
  biases = tf.Variable(tf.zeros([128]),
                       name="biases")
  hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
# Hidden 2
with tf.name_scope("hidden2"):
  weights = tf.Variable(
      tf.truncated_normal([128, 32],
                          stddev=1.0 / math.sqrt(float(128))),
      name="weights")
  biases = tf.Variable(tf.zeros([32]),
                       name="biases")
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
# Linear
with tf.name_scope("softmax_linear"):
  weights = tf.Variable(
      tf.truncated_normal([32, 10],
                          stddev=1.0 / math.sqrt(float(32))),
      name="weights")
  biases = tf.Variable(tf.zeros([10]),
                       name="biases")
  logits = tf.matmul(hidden2, weights) + biases
  tf.add_to_collection("logits", logits)

init_all_op = tf.global_variables_initializer()

with tf.Session() as sess:
  # Initializes all the variables.
  sess.run(init_all_op)
  # Runs to logit.
  sess.run(logits)
  # Creates a saver.
  saver0 = tf.train.Saver()
  saver0.save(sess, 'my-save-dir/my-model-10000')
  # Generates MetaGraphDef.
  saver0.export_meta_graph('my-save-dir/my-model-10000.meta')


# Then later import it and extend it to a training graph.
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # Addes loss and train.
  labels = tf.constant(0, tf.int32, shape=[100], name="labels")
  batch_size = tf.size(labels)
  logits = tf.get_collection("logits")[0]
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits)

  tf.summary.scalar('loss', loss)
  # Creates the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(0.01)

  # Runs train_op.
  train_op = optimizer.minimize(loss)
  sess.run(train_op)
