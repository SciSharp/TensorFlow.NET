using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class ConstantTest : EagerModeTestBase
    {
        Status status = new Status();

        [TestMethod]
        public void ScalarConst()
        {
            var tensor1 = tf.constant(8); // int
            Assert.AreEqual(tensor1.dtype, TF_DataType.TF_INT32);
            var tensor2 = tf.constant(6.0f); // float
            Assert.AreEqual(tensor2.dtype, TF_DataType.TF_FLOAT);
            var tensor3 = tf.constant(6.0); // double
            Assert.AreEqual(tensor3.dtype, TF_DataType.TF_DOUBLE);
        }

        /*[DataTestMethod]
        [DataRow(int.MinValue)]
        [DataRow(-1)]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(int.MaxValue)]
        public void ScalarConstTypecast_int(int value)
        {
            var tensor = (Tensor)value;
            with(tf.Session(), sess =>
            {
                var result = sess.run(tensor);
                Assert.AreEqual(result.Data<int>()[0], value);
            });
        }

        [DataTestMethod]
        [DataRow(double.NegativeInfinity)]
        [DataRow(double.MinValue)]
        [DataRow(-1d)]
        [DataRow(0d)]
        [DataRow(double.Epsilon)]
        [DataRow(1d)]
        [DataRow(double.MaxValue)]
        [DataRow(double.PositiveInfinity)]
        [DataRow(double.NaN)]
        public void ScalarConstTypecast_double(double value)
        {
            var tensor = (Tensor)value;
            with(tf.Session(), sess =>
            {
                var result = sess.run(tensor);
                Assert.AreEqual(result.Data<double>()[0], value);
            });
        }

        [DataTestMethod]
        [DataRow(float.NegativeInfinity)]
        [DataRow(float.MinValue)]
        [DataRow(-1f)]
        [DataRow(0f)]
        [DataRow(float.Epsilon)]
        [DataRow(1f)]
        [DataRow(float.MaxValue)]
        [DataRow(float.PositiveInfinity)]
        [DataRow(float.NaN)]
        public void ScalarConstTypecast_float(float value)
        {
            var tensor = (Tensor)value;
            with(tf.Session(), sess =>
            {
                var result = sess.run(tensor);
                Assert.AreEqual(result.Data<double>()[0], value);
            });
        }

        [TestMethod]
        public void StringConst()
        {
            string str = "Hello, TensorFlow.NET!";
            var tensor = tf.constant(str);
            with(tf.Session(), sess =>
            {
                var result = sess.run(tensor);
                Assert.IsTrue(result.Data<string>()[0] == str);
            });
        }*/

        [TestMethod]
        public void ZerosConst()
        {
            // small size
            var tensor = tf.zeros((3, 2), tf.int32, "small");

            Assert.AreEqual(tensor.shape[0], 3);
            Assert.AreEqual(tensor.shape[1], 2);
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0, 0, 0 }, tensor.numpy().ToArray<int>()));

            // big size
            tensor = tf.zeros((200, 100), tf.int32, "big");

            Assert.AreEqual(tensor.shape[0], 200);
            Assert.AreEqual(tensor.shape[1], 100);

            var data = tensor.numpy().ToArray<int>();
            Assert.AreEqual(0, data[0]);
            Assert.AreEqual(0, data[500]);
            Assert.AreEqual(0, data[data.Length - 1]);
        }

        [TestMethod]
        public void OnesConst()
        {
            var ones = tf.ones(new Shape(3, 2), tf.float32, "ones");
            Assert.AreEqual(ones.dtype, tf.float32);
            Assert.AreEqual(ones.shape[0], 3);
            Assert.AreEqual(ones.shape[1], 2);
            Assert.IsTrue(new float[] { 1, 1, 1, 1, 1, 1 }.SequenceEqual(ones.numpy().ToArray<float>()));
        }

        [TestMethod]
        public void OnesToHalves()
        {
            var ones = tf.ones(new Shape(3, 2), tf.float64, "ones");
            var halfes = ones * 0.5;
            Assert.AreEqual(halfes.shape[0], 3);
            Assert.AreEqual(halfes.shape[1], 2);
            Assert.IsTrue(new[] { .5, .5, .5, .5, .5, .5 }.SequenceEqual(halfes.numpy().ToArray<double>()));
        }

        [TestMethod]
        public void NDimConst()
        {
            var nd = np.array(new int[][]
            {
                new int[]{ 3, 1, 1 },
                new int[]{ 2, 1, 3 }
            });

            var tensor = tf.constant(nd);
            var data = tensor.numpy().ToArray<int>();

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 2, 3 }, tensor.shape));
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 1, 1, 2, 1, 3 }, data));
        }

        [TestMethod]
        public void Multiply()
        {
            var a = tf.constant(3.0);
            var b = tf.constant(2.0);
            var c = a * b;

            Assert.AreEqual(6.0, (double)c);
        }

        [TestMethod]
        public void Reshape()
        {
            var ones = tf.ones((3, 2), tf.float32, "ones");
            var reshaped = tf.reshape(ones, (2, 3));
            Assert.AreEqual(reshaped.dtype, tf.float32);
            Assert.AreEqual(reshaped.shape[0], 2);
            Assert.AreEqual(reshaped.shape[1], 3);
            Assert.IsTrue(new float[] { 1, 1, 1, 1, 1, 1 }.SequenceEqual(ones.numpy().ToArray<float>()));
        }
    }
}
