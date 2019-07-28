/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.ImageProcess
{
    /// <summary>
    /// Convolutional Neural Network classifier for Hand Written Digits
    /// CNN architecture with two convolutional layers, followed by two fully-connected layers at the end.
    /// Use Stochastic Gradient Descent (SGD) optimizer. 
    /// http://www.easy-tensorflow.com/tf-tutorials/convolutional-neural-nets-cnns/cnn1
    /// </summary>
    public class DigitRecognitionCNN : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "MNIST CNN";

        string logs_path = "logs";

        const int img_h = 28, img_w = 28; // MNIST images are 28x28
        int n_classes = 10; // Number of classes, one class per digit
        int n_channels = 1;

        // Hyper-parameters
        int epochs = 5; // accuracy > 98%
        int batch_size = 100;
        float learning_rate = 0.001f;
        Datasets<DataSetMnist> mnist;

        // Network configuration
        // 1st Convolutional Layer
        int filter_size1 = 5;  // Convolution filters are 5 x 5 pixels.
        int num_filters1 = 16; //  There are 16 of these filters.
        int stride1 = 1;  // The stride of the sliding window

        // 2nd Convolutional Layer
        int filter_size2 = 5; // Convolution filters are 5 x 5 pixels.
        int num_filters2 = 32;// There are 32 of these filters.
        int stride2 = 1;  // The stride of the sliding window

        // Fully-connected layer.
        int h1 = 128; // Number of neurons in fully-connected layer.

        Tensor x, y;
        Tensor loss, accuracy, cls_prediction;
        Operation optimizer;

        int display_freq = 100;
        float accuracy_test = 0f;
        float loss_test = 1f;

        NDArray x_train, y_train;
        NDArray x_valid, y_valid;
        NDArray x_test, y_test;

        public bool Run()
        {
            PrepareData();
            BuildGraph();

            with(tf.Session(), sess =>
            {
                Train(sess);
                Test(sess);
            });

            return loss_test < 0.05 && accuracy_test > 0.98;
        }

        public Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            with(tf.name_scope("Input"), delegate
            {
                // Placeholders for inputs (x) and outputs(y)
                x = tf.placeholder(tf.float32, shape: (-1, img_h, img_w, n_channels), name: "X");
                y = tf.placeholder(tf.float32, shape: (-1, n_classes), name: "Y");
            });

            var conv1 = conv_layer(x, filter_size1, num_filters1, stride1, name: "conv1");
            var pool1 = max_pool(conv1, ksize: 2, stride: 2, name: "pool1");
            var conv2 = conv_layer(pool1, filter_size2, num_filters2, stride2, name: "conv2");
            var pool2 = max_pool(conv2, ksize: 2, stride: 2, name: "pool2");
            var layer_flat = flatten_layer(pool2);
            var fc1 = fc_layer(layer_flat, h1, "FC1", use_relu: true);
            var output_logits = fc_layer(fc1, n_classes, "OUT", use_relu: false);

            with(tf.variable_scope("Train"), delegate
            {
                with(tf.variable_scope("Loss"), delegate
                {
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: y, logits: output_logits), name: "loss");
                });

                with(tf.variable_scope("Optimizer"), delegate
                {
                    optimizer = tf.train.AdamOptimizer(learning_rate: learning_rate, name: "Adam-op").minimize(loss);
                });

                with(tf.variable_scope("Accuracy"), delegate
                {
                    var correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name: "correct_pred");
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name: "accuracy");
                });

                with(tf.variable_scope("Prediction"), delegate
                {
                    cls_prediction = tf.argmax(output_logits, axis: 1, name: "predictions");
                });
            });

            return graph;
        }

        public void Train(Session sess)
        {
            // Number of training iterations in each epoch
            var num_tr_iter = y_train.len / batch_size;

            var init = tf.global_variables_initializer();
            sess.run(init);

            float loss_val = 100.0f;
            float accuracy_val = 0f;

            foreach (var epoch in range(epochs))
            {
                print($"Training epoch: {epoch + 1}");
                // Randomly shuffle the training data at the beginning of each epoch 
                (x_train, y_train) = mnist.Randomize(x_train, y_train);

                foreach (var iteration in range(num_tr_iter))
                {
                    var start = iteration * batch_size;
                    var end = (iteration + 1) * batch_size;
                    var (x_batch, y_batch) = mnist.GetNextBatch(x_train, y_train, start, end);

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
                var results1 = sess.run(new[] { loss, accuracy }, new FeedItem(x, x_valid), new FeedItem(y, y_valid));
                loss_val = results1[0];
                accuracy_val = results1[1];
                print("---------------------------------------------------------");
                print($"Epoch: {epoch + 1}, validation loss: {loss_val.ToString("0.0000")}, validation accuracy: {accuracy_val.ToString("P")}");
                print("---------------------------------------------------------");
            }
        }

        public void Test(Session sess)
        {
            var result = sess.run(new[] { loss, accuracy }, new FeedItem(x, x_test), new FeedItem(y, y_test));
            loss_test = result[0];
            accuracy_test = result[1];
            print("---------------------------------------------------------");
            print($"Test loss: {loss_test.ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}");
            print("---------------------------------------------------------");
        }

        /// <summary>
        /// Create a 2D convolution layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="filter_size">size of each filter</param>
        /// <param name="num_filters">number of filters(or output feature maps)</param>
        /// <param name="stride">filter stride</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor conv_layer(Tensor x, int filter_size, int num_filters, int stride, string name)
        {
            return with(tf.variable_scope(name), delegate {

                var num_in_channel = x.shape[x.NDims - 1];
                var shape = new[] { filter_size, filter_size, num_in_channel, num_filters };
                var W = weight_variable("W", shape);
                // var tf.summary.histogram("weight", W);
                var b = bias_variable("b", new[] { num_filters });
                // tf.summary.histogram("bias", b);
                var layer = tf.nn.conv2d(x, W,
                                     strides: new[] { 1, stride, stride, 1 },
                                     padding: "SAME");
                layer += b;
                return tf.nn.relu(layer);
            });
        }

        /// <summary>
        /// Create a max pooling layer
        /// </summary>
        /// <param name="x">input to max-pooling layer</param>
        /// <param name="ksize">size of the max-pooling filter</param>
        /// <param name="stride">stride of the max-pooling filter</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor max_pool(Tensor x, int ksize, int stride, string name)
        {
            return tf.nn.max_pool(x,
                ksize: new[] { 1, ksize, ksize, 1 },
                strides: new[] { 1, stride, stride, 1 },
                padding: "SAME",
                name: name);
        }

        /// <summary>
        /// Flattens the output of the convolutional layer to be fed into fully-connected layer
        /// </summary>
        /// <param name="layer">input array</param>
        /// <returns>flattened array</returns>
        private Tensor flatten_layer(Tensor layer)
        {
            return with(tf.variable_scope("Flatten_layer"), delegate
            {
                var layer_shape = layer.TensorShape;
                var num_features = layer_shape[new Slice(1, 4)].Size;
                var layer_flat = tf.reshape(layer, new[] { -1, num_features });

                return layer_flat;
            });
        }

        /// <summary>
        /// Create a weight variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private RefVariable weight_variable(string name, int[] shape)
        {
            var initer = tf.truncated_normal_initializer(stddev: 0.01f);
            return tf.get_variable(name,
                                   dtype: tf.float32,
                                   shape: shape,
                                   initializer: initer);
        }

        /// <summary>
        /// Create a bias variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private RefVariable bias_variable(string name, int[] shape)
        {
            var initial = tf.constant(0f, shape: shape, dtype: tf.float32);
            return tf.get_variable(name,
                           dtype: tf.float32,
                           initializer: initial);
        }

        /// <summary>
        /// Create a fully-connected layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="num_units">number of hidden units in the fully-connected layer</param>
        /// <param name="name">layer name</param>
        /// <param name="use_relu">boolean to add ReLU non-linearity (or not)</param>
        /// <returns>The output array</returns>
        private Tensor fc_layer(Tensor x, int num_units, string name, bool use_relu = true)
        {
            return with(tf.variable_scope(name), delegate
            {
                var in_dim = x.shape[1];

                var W = weight_variable("W_" + name, shape: new[] { in_dim, num_units });
                var b = bias_variable("b_" + name, new[] { num_units });

                var layer = tf.matmul(x, W) + b;
                if (use_relu)
                    layer = tf.nn.relu(layer);

                return layer;
            });
        } 
            
        public void PrepareData()
        {
            mnist = MNIST.read_data_sets("mnist", one_hot: true);
            (x_train, y_train) = Reformat(mnist.train.data, mnist.train.labels);
            (x_valid, y_valid) = Reformat(mnist.validation.data, mnist.validation.labels);
            (x_test, y_test) = Reformat(mnist.test.data, mnist.test.labels);

            print("Size of:");
            print($"- Training-set:\t\t{len(mnist.train.data)}");
            print($"- Validation-set:\t{len(mnist.validation.data)}");
        }

        /// <summary>
        /// Reformats the data to the format acceptable for convolutional layers
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        private (NDArray, NDArray) Reformat(NDArray x, NDArray y)
        {
            var (img_size, num_ch, num_class) = (np.sqrt(x.shape[1]), 1, len(np.unique<int>(np.argmax(y, 1))));
            var dataset = x.reshape(x.shape[0], img_size, img_size, num_ch).astype(np.float32);
            //y[0] = np.arange(num_class) == y[0];
            //var labels = (np.arange(num_class) == y.reshape(y.shape[0], 1, y.shape[1])).astype(np.float32);
            return (dataset, y);
        }

        public Graph ImportGraph() => throw new NotImplementedException();

        public void Predict(Session sess) => throw new NotImplementedException();
    }
}
