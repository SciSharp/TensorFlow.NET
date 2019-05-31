using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A logistic regression learning algorithm example using TensorFlow library.
    /// This example is using the MNIST database of handwritten digits
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
    /// </summary>
    public class LogisticRegression : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Logistic Regression";
        public bool IsImportingGraph { get; set; } = false;


        public int training_epochs = 10;
        public int? train_size = null;
        public int validation_size = 5000;
        public int? test_size = null;
        public int batch_size = 100;

        private float learning_rate = 0.01f;
        private int display_step = 1;

        Datasets mnist;

        public bool Run()
        {
            PrepareData();

            // tf Graph Input
            var x = tf.placeholder(tf.float32, new TensorShape(-1, 784)); // mnist data image of shape 28*28=784
            var y = tf.placeholder(tf.float32, new TensorShape(-1, 10)); // 0-9 digits recognition => 10 classes

            // Set model weights
            var W = tf.Variable(tf.zeros(new Shape(784, 10)));
            var b = tf.Variable(tf.zeros(new Shape(10)));

            // Construct model
            var pred = tf.nn.softmax(tf.matmul(x, W) + b); // Softmax

            // Minimize error using cross entropy
            var cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices: 1));

            // Gradient Descent
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            var sw = new Stopwatch();

            return with(tf.Session(), sess =>
            {
                // Run the initializer
                sess.run(init);

                // Training cycle
                foreach (var epoch in range(training_epochs))
                {
                    sw.Start();

                    var avg_cost = 0.0f;
                    var total_batch = mnist.train.num_examples / batch_size;
                    // Loop over all batches
                    foreach (var i in range(total_batch))
                    {
                        var (batch_xs, batch_ys) = mnist.train.next_batch(batch_size);
                        // Run optimization op (backprop) and cost op (to get loss value)
                        var result = sess.run(new object[] { optimizer, cost },
                            new FeedItem(x, batch_xs),
                            new FeedItem(y, batch_ys));

                        float c = result[1];
                        // Compute average loss
                        avg_cost += c / total_batch;
                    }

                    sw.Stop();

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                        print($"Epoch: {(epoch + 1).ToString("D4")} Cost: {avg_cost.ToString("G9")} Elapse: {sw.ElapsedMilliseconds}ms");

                    sw.Reset();
                }

                print("Optimization Finished!");
                // SaveModel(sess);

                // Test model
                var correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
                // Calculate accuracy
                var accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
                float acc = accuracy.eval(new FeedItem(x, mnist.test.images), new FeedItem(y, mnist.test.labels));
                print($"Accuracy: {acc.ToString("F4")}");

                return acc > 0.9;
            });
        }

        public void PrepareData()
        {
            mnist = MnistDataSet.read_data_sets("mnist", one_hot: true, train_size: train_size, validation_size: validation_size, test_size: test_size);
        }

        public void SaveModel(Session sess)
        {
            var saver = tf.train.Saver();
            var save_path = saver.save(sess, "logistic_regression/model.ckpt");
            tf.train.write_graph(sess.graph, "logistic_regression", "model.pbtxt", as_text: true);

            FreezeGraph.freeze_graph(input_graph: "logistic_regression/model.pbtxt",
                              input_saver: "",
                              input_binary: false,
                              input_checkpoint: "logistic_regression/model.ckpt",
                              output_node_names: "Softmax",
                              restore_op_name: "save/restore_all",
                              filename_tensor_name: "save/Const:0",
                              output_graph: "logistic_regression/model.pb",
                              clear_devices: true,
                              initializer_nodes: "");
        }

        public void Predict()
        {
            var graph = new Graph().as_default();
            graph.Import(Path.Join("logistic_regression", "model.pb"));

            with(tf.Session(graph), sess =>
            {
                // restoring the model
                // var saver = tf.train.import_meta_graph("logistic_regression/tensorflowModel.ckpt.meta");
                // saver.restore(sess, tf.train.latest_checkpoint('logistic_regression'));
                var pred = graph.OperationByName("Softmax");
                var output = pred.outputs[0];
                var x = graph.OperationByName("Placeholder");
                var input = x.outputs[0];

                // predict
                var (batch_xs, batch_ys) = mnist.train.next_batch(10);
                var results = sess.run(output, new FeedItem(input, batch_xs[np.arange(1)]));

                if (results.argmax() == (batch_ys[0] as NDArray).argmax())
                    print("predicted OK!");
                else
                    throw new ValueError("predict error, should be 90% accuracy");
            });
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public bool Train()
        {
            throw new NotImplementedException();
        }

        bool IExample.Predict()
        {
            throw new NotImplementedException();
        }
    }
}
