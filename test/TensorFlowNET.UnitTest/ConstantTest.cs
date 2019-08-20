using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class ConstantTest
    {
        Status status = new Status();

        [TestMethod]
        public void ScalarConst()
        {
            var tensor1 = tf.constant(8); // int
            var tensor2 = tf.constant(6.0f); // float
            var tensor3 = tf.constant(6.0); // double
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
            var tensor = tf.zeros(new Shape(3, 2), TF_DataType.TF_INT32, "small");
            using (var sess = tf.Session())
            {
                var result = sess.run(tensor);

                Assert.AreEqual(result[0].shape[0], 3);
                Assert.AreEqual(result[0].shape[1], 2);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0, 0, 0 }, result[0].Data<int>()));
            }

            // big size
            tensor = tf.zeros(new Shape(200, 100), TF_DataType.TF_INT32, "big");
            using (var sess = tf.Session())
            {
                var result = sess.run(tensor);

                Assert.AreEqual(result[0].shape[0], 200);
                Assert.AreEqual(result[0].shape[1], 100);

                var data = result[0].Data<int>();
                Assert.AreEqual(0, data[0]);
                Assert.AreEqual(0, data[500]);
                Assert.AreEqual(0, data[result[0].size - 1]);
            }
        }

        [TestMethod]
        public void OnesConst()
        {
            var ones = tf.ones(new Shape(3, 2), TF_DataType.TF_DOUBLE, "ones");
            using (var sess = tf.Session())
            {
                var result = sess.run(ones);

                Assert.AreEqual(result[0].shape[0], 3);
                Assert.AreEqual(result[0].shape[1], 2);
                Assert.IsTrue(new[] { 1, 1, 1, 1, 1, 1 }.SequenceEqual(result[0].Data<int>()));
            }
        }

        [TestMethod]
        public void OnesToHalves()
        {
            var ones = tf.ones(new Shape(3, 2), TF_DataType.TF_DOUBLE, "ones");
            var halfes = ones * 0.5;
            using (var sess = tf.Session())
            {
                var result = sess.run(halfes);

                Assert.AreEqual(result[0].shape[0], 3);
                Assert.AreEqual(result[0].shape[1], 2);
                Assert.IsTrue(new[] { .5, .5, .5, .5, .5, .5 }.SequenceEqual(result[0].Data<double>()));
            }
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
            using (var sess = tf.Session())
            {
                var result = sess.run(tensor);
                var data = result[0].Data<int>();

                Assert.AreEqual(result[0].shape[0], 2);
                Assert.AreEqual(result[0].shape[1], 3);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 1, 1, 2, 1, 3 }, data));
            }
        }

        [TestMethod]
        public void Multiply()
        {
            var a = tf.constant(3.0);
            var b = tf.constant(2.0);
            var c = a * b;

            var sess = tf.Session();
            double result = sess.run(c)[0];
            sess.close();

            Assert.AreEqual(6.0, result);
        }

        [TestMethod]
        public void StringEncode()
        {
            string str = "Hello, TensorFlow.NET!";
            var handle = Marshal.StringToHGlobalAnsi(str);
            ulong dst_len = (ulong)c_api.TF_StringEncodedSize((UIntPtr)str.Length);
            Assert.AreEqual(dst_len, (ulong)23);
            IntPtr dst = Marshal.AllocHGlobal((int)dst_len);
            ulong encoded_len = c_api.TF_StringEncode(handle, (ulong)str.Length, dst, dst_len, status);
            Assert.AreEqual((ulong)23, encoded_len);
            Assert.AreEqual(status.Code, TF_Code.TF_OK);
            string encoded_str = Marshal.PtrToStringUTF8(dst + sizeof(byte));
            Assert.AreEqual(encoded_str, str);
            Assert.AreEqual(str.Length, Marshal.ReadByte(dst));
            //c_api.TF_StringDecode(dst, (ulong)str.Length, IntPtr.Zero, ref dst_len, status);
        }

        /// <summary>
        /// tensorflow\c\c_api_test.cc
        /// TestEncodeDecode
        /// </summary>
        [TestMethod]
        public void EncodeDecode()
        {

        }
    }
}
