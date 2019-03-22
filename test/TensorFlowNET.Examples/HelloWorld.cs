using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

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
            /* Create a Constant op
               The op is added as a node to the default graph.
            
               The value returned by the constructor represents the output
               of the Constant op. */
            var str = "Hello, TensorFlow.NET!";
            var hello = tf.constant(str);

            // Start tf session
            using (var sess = tf.Session())
            {
                // Run the op
                var result = sess.run(hello);
                Console.WriteLine(result.ToString());
                if(!result.ToString().Equals(str))
                {
                    throw new ValueError("HelloWorld example acts in unexpected way.");
                }
            }
        }

        public void PrepareData()
        {
        }
    }
}
