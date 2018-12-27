using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class ConstantTest
    {
        Tensor tensor;

        [TestMethod]
        public void ScalarConst()
        {
            tensor = tf.constant(8); // int
            tensor = tf.constant(6.0f); // float
            tensor = tf.constant(6.0); // double
        }

        [TestMethod]
        public void StringConst()
        {
            tensor = tf.constant("Elephant");
        }
    }
}
