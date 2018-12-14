using System;
using System.Collections.Generic;
using System.Text;
using tf = TensorFlowNET.Core.Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Simple hello world using TensorFlow
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/helloworld.py
    /// </summary>
    public class HelloWorld : IExample
    {
        public void Run()
        {
            /* # Create a Constant op
              The op is added as a node to the default graph.
            
              The value returned by the constructor represents the output
              of the Constant op.*/
            var hello = tf.constant("Hello, TensorFlow!");

            // Start tf session
            var sess = tf.Session();

            // Run the op
            sess.run(hello);
        }
    }
}
