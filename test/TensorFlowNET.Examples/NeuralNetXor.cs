using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Simple vanilla neural net solving the famous XOR problem
    /// https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/getting_started/xor/README.md
    /// </summary>
    public class NeuralNetXor : Python, IExample
    {
        public int Priority => 2;
        public bool Enabled { get; set; } = true;
        public string Name => "NN XOR";

        public int num_steps = 5000;

        private (Operation, Tensor, RefVariable) make_graph(Tensor features,Tensor labels, int num_hidden = 8)
        {
            var stddev = 1 / Math.Sqrt(2);
            var hidden_weights = tf.Variable(tf.truncated_normal(new int []{2, num_hidden}, stddev: (float) stddev ));

            // Shape [4, num_hidden]
            var hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));

            var output_weights = tf.Variable(tf.truncated_normal(
                new[] {num_hidden, 1},
                stddev: (float) (1 / Math.Sqrt(num_hidden))
            ));

            // Shape [4, 1]
            var logits = tf.matmul(hidden_activations, output_weights);

            // Shape [4]
            var predictions = tf.sigmoid(tf.squeeze(logits));
            var loss = tf.reduce_mean(tf.square(predictions - tf.cast(labels, tf.float32)));

            var gs = tf.Variable(0, trainable: false);
            var train_op = tf.train.GradientDescentOptimizer(0.2f).minimize(loss, global_step: gs);

            return (train_op, loss, gs);
        }

        public bool Run()
        {

            var graph = tf.Graph();

            var init=with(graph.as_default(), g =>
            {
                var features = tf.placeholder(tf.float32, new TensorShape(4, 2));
                var labels = tf.placeholder(tf.int32, new TensorShape(4));

                var (train_op, loss, gs) = make_graph(features, labels);
                return tf.global_variables_initializer();
            });

            // Start tf session
            with<Session>(tf.Session(), sess =>
            {
                init.run();
                var step = 0;
                var xy = np.array(new bool[,]
                {
                    {true, false, },
                    {true, true, },
                    {false, false, },
                    {false, true, },
                }, dtype: np.float32);

                var y_ = np.array(new[] {true, false, false, true}, dtype: np.int32);
                while (step < num_steps)
                {
                    // original python:
                    //_, step, loss_value = sess.run(
                    //          [train_op, gs, loss],
                    //          feed_dict={features: xy, labels: y_}
                    //      )
                    // TODO: how the hell to port that to c#?
                    //  var ( _, step, loss_value) = sess.run(new object[] {train_op, gs, loss},feed_dict: new {"features": xy, "labels": y_});
                }
                //tf.logging.info('Final loss is: {}'.format(loss_value))
                //Console.WriteLine($"Final loss is: {loss_value}");

            });
            return true;
        }

        public void PrepareData()
        {
        }
    }
}
