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
        public unsafe void NewTensor()
        {
            var nd = np.array(1f, 2f, 3f, 4f, 5f, 6f).reshape(2, 3);

            var tensor = new Tensor(nd);
            var array = tensor.Data<float>();

            Assert.AreEqual(tensor.dtype, TF_DataType.TF_FLOAT);
            Assert.AreEqual(tensor.rank, nd.ndim);
            Assert.AreEqual(tensor.shape[0], nd.shape[0]);
            Assert.AreEqual(tensor.shape[1], nd.shape[1]);
            Assert.AreEqual(tensor.bytesize, (uint)nd.size * sizeof(float));
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<float>(), array));
        }
    }
}
