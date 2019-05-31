using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Simple vanilla neural net solving the famous XOR problem
    /// https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/getting_started/xor/README.md
    /// </summary>
    public class NeuralNetXor : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "NN XOR";
        public bool IsImportingGraph { get; set; } = false;

        public int num_steps = 10000;

        private NDArray data;

        private (Operation, Tensor, Tensor) make_graph(Tensor features,Tensor labels, int num_hidden = 8)
        {
            var stddev = 1 / Math.Sqrt(2);
            var hidden_weights = tf.Variable(tf.truncated_normal(new int []{2, num_hidden}, seed:1, stddev: (float) stddev ));

            // Shape [4, num_hidden]
            var hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));

            var output_weights = tf.Variable(tf.truncated_normal(
                new[] {num_hidden, 1},
                seed: 17,
                stddev: (float) (1 / Math.Sqrt(num_hidden))
            ));

            // Shape [4, 1]
            var logits = tf.matmul(hidden_activations, output_weights);

            // Shape [4]
            var predictions = tf.sigmoid(tf.squeeze(logits));
            var loss = tf.reduce_mean(tf.square(predictions - tf.cast(labels, tf.float32)), name:"loss");

            var gs = tf.Variable(0, trainable: false, name: "global_step");
            var train_op = tf.train.GradientDescentOptimizer(0.2f).minimize(loss, global_step: gs);

            return (train_op, loss, gs);
        }

        public bool Run()
        {
            PrepareData();
            float loss_value = 0;
            if (IsImportingGraph)
                loss_value = RunWithImportedGraph();
            else
                loss_value = RunWithBuiltGraph();

            return loss_value < 0.0628;
        }

        private float RunWithImportedGraph()
        {
            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("graph/xor.meta");

            Tensor features = graph.get_operation_by_name("Placeholder");
            Tensor labels = graph.get_operation_by_name("Placeholder_1");
            Tensor loss = graph.get_operation_by_name("loss");
            Tensor train_op = graph.get_operation_by_name("train_op");
            Tensor global_step = graph.get_operation_by_name("global_step");

            var init = tf.global_variables_initializer();
            float loss_value = 0;
            // Start tf session
            with(tf.Session(graph), sess =>
            {
                sess.run(init);
                var step = 0;

                var y_ = np.array(new int[] { 1, 0, 0, 1 }, dtype: np.int32);
                while (step < num_steps)
                {
                    // original python:
                    //_, step, loss_value = sess.run(
                    //          [train_op, gs, loss],
                    //          feed_dict={features: xy, labels: y_}
                    //      )
                    var result = sess.run(new ITensorOrOperation[] { train_op, global_step, loss }, new FeedItem(features, data), new FeedItem(labels, y_));
                    loss_value = result[2];
                    step = result[1];
                    if (step % 1000 == 0)
                        Console.WriteLine($"Step {step} loss: {loss_value}");
                }
                Console.WriteLine($"Final loss: {loss_value}");
            });

            return loss_value;
        }

        private float RunWithBuiltGraph()
        {
            var graph = tf.Graph().as_default();

            var features = tf.placeholder(tf.float32, new TensorShape(4, 2));
            var labels = tf.placeholder(tf.int32, new TensorShape(4));

            var (train_op, loss, gs) = make_graph(features, labels);

            var init = tf.global_variables_initializer();

            float loss_value = 0;
            // Start tf session
            with(tf.Session(graph), sess =>
            {
                sess.run(init);
                var step = 0;

                var y_ = np.array(new int[] { 1, 0, 0, 1 }, dtype: np.int32);
                while (step < num_steps)
                {
                    var result = sess.run(new ITensorOrOperation[] { train_op, gs, loss }, new FeedItem(features, data), new FeedItem(labels, y_));
                    loss_value = result[2];
                    step = result[1];
                    if (step % 1000 == 0)
                        Console.WriteLine($"Step {step} loss: {loss_value}");
                }
                Console.WriteLine($"Final loss: {loss_value}");
            });

            return loss_value;
        }

        public void PrepareData()
        {
            data = new float[,]
            {
                {1, 0 },
                {1, 1 },
                {0, 0 },
                {0, 1 }
            };

            if (IsImportingGraph)
            {
                // download graph meta data
                string url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/xor.meta";
                Web.Download(url, "graph", "xor.meta");
            }
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

        public bool Predict()
        {
            throw new NotImplementedException();
        }
    }
}
