using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
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
        public void ZerosConst()
        {
            tensor = tf.zeros(new Shape(3, 2), TF_DataType.TF_INT32, "x");
            Assert.AreEqual(tensor.shape[0], 3);
            Assert.AreEqual(tensor.shape[0], 2);
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0, 0, 0 }, tensor.Data<int>()));

            tensor = tf.zeros(new Shape(200, 300), TF_DataType.TF_INT32, "x");
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
