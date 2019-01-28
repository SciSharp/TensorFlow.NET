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
            string str = "Hello, TensorFlow.NET!";
            tensor = tf.constant(str);
            Python.with<Session>(tf.Session(), sess =>
            {
                var result = sess.run(tensor);
                Assert.IsTrue(result.Data<string>()[0] == str);
            });
        }

        [TestMethod]
        public void ZerosConst()
        {
            // small size
            tensor = tf.zeros(new Shape(3, 2), TF_DataType.TF_INT32, "small");
            Python.with<Session>(tf.Session(), sess =>
            {
                var result = sess.run(tensor);

                Assert.AreEqual(result.shape[0], 3);
                Assert.AreEqual(result.shape[1], 2);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0, 0, 0 }, result.Data<int>()));
            });

            // big size
            tensor = tf.zeros(new Shape(200, 100), TF_DataType.TF_INT32, "big");
            Python.with<Session>(tf.Session(), sess =>
            {
                var result = sess.run(tensor);

                Assert.AreEqual(result.shape[0], 200);
                Assert.AreEqual(result.shape[1], 100);

                var data = result.Data<int>();
                Assert.AreEqual(0, data[0]);
                Assert.AreEqual(0, data[result.size - 1]);
            });
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
