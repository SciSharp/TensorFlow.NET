using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Basic Operations example using TensorFlow library.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_operations.py
    /// </summary>
    public class BasicOperations : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Basic Operations";
        public bool IsImportingGraph { get; set; } = false;

        private Session sess;

        public bool Run()
        {
            // Basic constant operations
            // The value returned by the constructor represents the output
            // of the Constant op.
            var a = tf.constant(2);
            var b = tf.constant(3);
            
            // Launch the default graph.
            using (sess = tf.Session())
            {
                Console.WriteLine("a=2, b=3");
                Console.WriteLine($"Addition with constants: {sess.run(a + b)}");
                Console.WriteLine($"Multiplication with constants: {sess.run(a * b)}");
            }

            // Basic Operations with variable as graph input
            // The value returned by the constructor represents the output
            // of the Variable op. (define as input when running session)
            // tf Graph input
            a = tf.placeholder(tf.int16);
            b = tf.placeholder(tf.int16);

            // Define some operations
            var add = tf.add(a, b);
            var mul = tf.multiply(a, b);

            // Launch the default graph.
            using(sess = tf.Session())
            {
                var feed_dict = new FeedItem[]
                {
                    new FeedItem(a, (short)2),
                    new FeedItem(b, (short)3)
                };
                // Run every operation with variable input
                Console.WriteLine($"Addition with variables: {sess.run(add, feed_dict)}");
                Console.WriteLine($"Multiplication with variables: {sess.run(mul, feed_dict)}");
            }

            // ----------------
            // More in details:
            // Matrix Multiplication from TensorFlow official tutorial

            // Create a Constant op that produces a 1x2 matrix.  The op is
            // added as a node to the default graph.
            //
            // The value returned by the constructor represents the output
            // of the Constant op.
            var nd1 = np.array(3, 3).reshape(1, 2);
            var matrix1 = tf.constant(nd1);

            // Create another Constant that produces a 2x1 matrix.
            var nd2 = np.array(2, 2).reshape(2, 1);
            var matrix2 = tf.constant(nd2);

            // Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
            // The returned value, 'product', represents the result of the matrix
            // multiplication.
            var product = tf.matmul(matrix1, matrix2);

            // To run the matmul op we call the session 'run()' method, passing 'product'
            // which represents the output of the matmul op.  This indicates to the call
            // that we want to get the output of the matmul op back.
            //
            // All inputs needed by the op are run automatically by the session.  They
            // typically are run in parallel.
            //
            // The call 'run(product)' thus causes the execution of threes ops in the
            // graph: the two constants and matmul.
            //
            // The output of the op is returned in 'result' as a numpy `ndarray` object.
            return with(tf.Session(), sess =>
            {
                var result = sess.run(product);
                Console.WriteLine(result.ToString()); // ==> [[ 12.]]
                return result.Data<int>()[0] == 12;
            });
        }

        public void PrepareData()
        {
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
