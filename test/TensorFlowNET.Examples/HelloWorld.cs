using System;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Simple hello world using TensorFlow
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/helloworld.py
    /// </summary>
    public class HelloWorld : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Hello World";
        public bool IsImportingGraph { get; set; } = false;

        public bool Run()
        {
            /* Create a Constant op
               The op is added as a node to the default graph.
            
               The value returned by the constructor represents the output
               of the Constant op. */
            var str = "Hello, TensorFlow.NET!";
            var hello = tf.constant(str);

            // Start tf session
            return with(tf.Session(), sess =>
            {
                // Run the op
                var result = sess.run(hello);
                Console.WriteLine(result.ToString());
                return result.ToString().Equals(str);
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

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
