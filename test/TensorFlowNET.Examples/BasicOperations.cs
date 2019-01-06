using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Basic Operations example using TensorFlow library.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_operations.py
    /// </summary>
    public class BasicOperations : IExample
    {
        private Session sess;

        public void Run()
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
                var feed_dict = new Dictionary<Tensor, NDArray>();
                feed_dict.Add(a, (short)2);
                feed_dict.Add(b, (short)3);
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

        }
    }
}
