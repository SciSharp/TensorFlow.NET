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
    public class TensorTest
    {
        [TestMethod]
        public void TF_NewTensor()
        {
            var nd = np.array(1f, 2f, 3f, 4f, 5f, 6f).reshape(2, 3);

            var data = Marshal.AllocHGlobal(sizeof(float) * nd.size);
            Marshal.Copy(nd.Data<float>(), 0, data, nd.size);

            var handle = c_api.TF_NewTensor(TF_DataType.TF_FLOAT, 
                nd.shape.Select(x => (long)x).ToArray(), // shape
                nd.ndim,
                data, 
                (UIntPtr)(nd.size * sizeof(float)), 
                tf.FreeTensorData, 
                IntPtr.Zero);

            Assert.AreNotEqual(handle, IntPtr.Zero);

            var tensor = new Tensor(handle);

            Assert.AreEqual(tensor.dtype, TF_DataType.TF_FLOAT);
            Assert.AreEqual(tensor.ndim, nd.ndim);
            Assert.AreEqual(nd.shape[0], c_api.TF_Dim(handle, 0));
            Assert.AreEqual(nd.shape[1], c_api.TF_Dim(handle, 1));
            Assert.AreEqual(tensor.bytesize, (uint)nd.size * sizeof(float));

            // Column major order
            // https://en.wikipedia.org/wiki/File:Row_and_column_major_order.svg
            // matrix:[[1, 2, 3], [4, 5, 6]]
            // index:   0  2  4    1  3  5
            // result:  1  4  2    5  3  6
            var array = tensor.Data<float>();
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<float>(), array));
        }
    }
}
