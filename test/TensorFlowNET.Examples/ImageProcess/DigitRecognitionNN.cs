using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.ImageProcess
{
    /// <summary>
    /// Neural Network classifier for Hand Written Digits
    /// Sample Neural Network architecture with two layers implemented for classifying MNIST digits.
    /// Use Stochastic Gradient Descent (SGD) optimizer. 
    /// http://www.easy-tensorflow.com/tf-tutorials/neural-networks
    /// </summary>
    public class DigitRecognitionNN : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "Digits Recognition Neural Network";

        const int img_h = 28;
        const int img_w = 28;
        int img_size_flat = img_h * img_w; // 784, the total number of pixels
        int n_classes = 10; // Number of classes, one class per digit
        // Hyper-parameters
        int epochs = 10;
        int batch_size = 100;
        float learning_rate = 0.001f;
        int h1 = 200; // number of nodes in the 1st hidden layer
        Datasets mnist;

        Tensor x, y;
        Tensor loss, accuracy;
        Operation optimizer;

        int display_freq = 100;

        public bool Run()
        {
            PrepareData();
            BuildGraph();
            Train();

            return true;
        }

        public Graph BuildGraph()
        {
            var g = tf.Graph();

            // Placeholders for inputs (x) and outputs(y)
            x = tf.placeholder(tf.float32, shape: (-1, img_size_flat), name: "X");
            y = tf.placeholder(tf.float32, shape: (-1, n_classes), name: "Y");

            // Create a fully-connected layer with h1 nodes as hidden layer
            var fc1 = fc_layer(x, h1, "FC1", use_relu: true);
            // Create a fully-connected layer with n_classes nodes as output layer
            var output_logits = fc_layer(fc1, n_classes, "OUT", use_relu: false);
            // Define the loss function, optimizer, and accuracy
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels: y, logits: output_logits), name: "loss");
            optimizer = tf.train.AdamOptimizer(learning_rate: learning_rate, name: "Adam-op").minimize(loss);
            var correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name: "correct_pred");
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name: "accuracy");

            // Network predictions
            var cls_prediction = tf.argmax(output_logits, axis: 1, name: "predictions");

            return g;
        }

        private Tensor fc_layer(Tensor x, int num_units, string name, bool use_relu = true)
        {
            var in_dim = x.shape[1];

            var initer = tf.truncated_normal_initializer(stddev: 0.01f);
            var W = tf.get_variable("W_" + name,
                        dtype: tf.float32,
                        shape: (in_dim, num_units),
                        initializer: initer);

            var initial = tf.constant(0f, num_units);
            var b = tf.get_variable("b_" + name,
                        dtype: tf.float32,
                        initializer: initial);

            var layer = tf.matmul(x, W) + b;
            if (use_relu)
                layer = tf.nn.relu(layer);

            return layer;
        } 

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public bool Predict()
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            mnist = MnistDataSet.read_data_sets("mnist", one_hot: true);
        }

        public bool Train()
        {
            // Number of training iterations in each epoch
            var num_tr_iter = mnist.train.labels.len / batch_size;

            return with(tf.Session(), sess =>
            {
                var init = tf.global_variables_initializer();
                sess.run(init);

                float loss_val = 100.0f;
                float accuracy_val = 0f;

                foreach (var epoch in range(epochs))
                {
                    print($"Training epoch: {epoch + 1}");
                    // Randomly shuffle the training data at the beginning of each epoch 
                    var (x_train, y_train) = randomize(mnist.train.images, mnist.train.labels);

                    foreach (var iteration in range(num_tr_iter))
                    {
                        var start = iteration * batch_size;
                        var end = (iteration + 1) * batch_size;
                        var (x_batch, y_batch) = get_next_batch(x_train, y_train, start, end);

                        // Run optimization op (backprop)
                        sess.run(optimizer, new FeedItem(x, x_batch), new FeedItem(y, y_batch));

                        if (iteration % display_freq == 0)
                        {
                            // Calculate and display the batch loss and accuracy
                            var result = sess.run(new[] { loss, accuracy }, new FeedItem(x, x_batch), new FeedItem(y, y_batch));
                            loss_val = result[0];
                            accuracy_val = result[1];
                            print($"iter {iteration.ToString("000")}: Loss={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("P")}");
                        }
                    }

                    // Run validation after every epoch
                    var results1 = sess.run(new[] { loss, accuracy }, new FeedItem(x, mnist.validation.images), new FeedItem(y, mnist.validation.labels));
                    loss_val = results1[0];
                    accuracy_val = results1[1];
                    print("---------------------------------------------------------");
                    print($"Epoch: {epoch + 1}, validation loss: {loss_val.ToString("0.0000")}, validation accuracy: {accuracy_val.ToString("P")}");
                    print("---------------------------------------------------------");

                }

                return accuracy_val > 0.9;
            });
        }

        private (NDArray, NDArray) randomize(NDArray x, NDArray y)
        {
            var perm = np.random.permutation(y.shape[0]);

            np.random.shuffle(perm);
            return (mnist.train.images[perm], mnist.train.labels[perm]);
        }

        /// <summary>
        /// selects a few number of images determined by the batch_size variable (if you don't know why, read about Stochastic Gradient Method)
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        private (NDArray, NDArray) get_next_batch(NDArray x, NDArray y, int start, int end)
        {
            var x_batch = x[$"{start}:{end}"];
            var y_batch = y[$"{start}:{end}"];
            return (x_batch, y_batch);
        }
    }
}
