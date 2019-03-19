using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A logistic regression learning algorithm example using TensorFlow library.
    /// This example is using the MNIST database of handwritten digits
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
    /// </summary>
    public class LogisticRegression : Python, IExample
    {
        public void Run()
        {
            PrepareData();
        }

        private void PrepareData()
        {
            MnistDataSet.read_data_sets("logistic_regression", one_hot: true);
            
        }
    }
}
