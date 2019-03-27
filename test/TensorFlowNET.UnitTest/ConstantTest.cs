using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class ConstantTest : Python
    {
        Status status = new Status();

        [TestMethod]
        public void ScalarConst()
        {
            var tensor1 = tf.constant(8); // int
            var tensor2 = tf.constant(6.0f); // float
            var tensor3 = tf.constant(6.0); // double
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
        }

        [TestMethod]
        public void ZerosConst()
        {
            // small size
            var tensor = tf.zeros(new Shape(3, 2), TF_DataType.TF_INT32, "small");
            with(tf.Session(), sess =>
            {
                var result = sess.run(tensor);

                Assert.AreEqual(result.shape[0], 3);
                Assert.AreEqual(result.shape[1], 2);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0, 0, 0 }, result.Data<int>()));
            });

            // big size
            tensor = tf.zeros(new Shape(200, 100), TF_DataType.TF_INT32, "big");
            with(tf.Session(), sess =>
            {
                var result = sess.run(tensor);

                Assert.AreEqual(result.shape[0], 200);
                Assert.AreEqual(result.shape[1], 100);

                var data = result.Data<int>();
                Assert.AreEqual(0, data[0]);
                Assert.AreEqual(0, data[500]);
                Assert.AreEqual(0, data[result.size - 1]);
            });
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
            with(tf.Session(), sess =>
            {
                var result = sess.run(tensor);
                var data = result.Data<int>();

                Assert.AreEqual(result.shape[0], 2);
                Assert.AreEqual(result.shape[1], 3);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 1, 1, 2, 1, 3 }, data));
            });
        }

        [TestMethod]
        public void Multiply()
        {
            var a = tf.constant(3.0);
            var b = tf.constant(2.0);
            var c = a * b;

            var sess = tf.Session();
            double result = sess.run(c);
            sess.close();

            Assert.AreEqual(6.0, result);
        }

        [TestMethod]
        public void StringEncode()
        {
            /*string str = "Hello, TensorFlow.NET!";
            var handle = Marshal.StringToHGlobalAnsi(str);
            ulong dst_len = c_api.TF_StringEncodedSize((UIntPtr)str.Length);
            Assert.AreEqual(dst_len, (ulong)23);
            IntPtr dst = Marshal.AllocHGlobal((int)dst_len);
            ulong encoded_len = c_api.TF_StringEncode(handle, (ulong)str.Length, dst, dst_len, status);
            Assert.AreEqual((ulong)23, encoded_len);
            Assert.AreEqual(status.Code, TF_Code.TF_OK);
            string encoded_str = Marshal.PtrToStringUTF8(dst + sizeof(byte));
            Assert.AreEqual(encoded_str, str);
            Assert.AreEqual(str.Length, Marshal.ReadByte(dst));*/
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
