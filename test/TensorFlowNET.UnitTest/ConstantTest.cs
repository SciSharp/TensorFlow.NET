using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
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

        [TestMethod]
        public void NDimConst()
        {
            var nd = np.array(new int[][]
            {
                new int[]{ 1, 2, 3 },
                new int[]{ 4, 5, 6 }
            });

            tensor = tf.constant(nd);
        }
    }
}
