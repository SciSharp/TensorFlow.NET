using System;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Basic introduction to TensorFlow's Eager API.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_eager_api.py
    /// </summary>
    public class BasicEagerApi : IExample
    {
        public bool Enabled { get; set; } = false;
        public string Name => "Basic Eager";
        public bool IsImportingGraph { get; set; } = false;

        private Tensor a, b, c, d;

        public bool Run()
        {
            // Set Eager API
            Console.WriteLine("Setting Eager mode...");
            tf.enable_eager_execution();

            // Define constant tensors
            Console.WriteLine("Define constant tensors");
            a = tf.constant(2);
            Console.WriteLine($"a = {a}");
            b = tf.constant(3);
            Console.WriteLine($"b = {b}");

            // Run the operation without the need for tf.Session
            Console.WriteLine("Running operations, without tf.Session");
            c = a + b;
            Console.WriteLine($"a + b = {c}");
            d = a * b;
            Console.WriteLine($"a * b = {d}");

            // Full compatibility with Numpy

            return true;
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

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
