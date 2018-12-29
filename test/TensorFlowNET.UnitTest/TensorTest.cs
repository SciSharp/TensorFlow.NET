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
        public void NewTensor()
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

        /// <summary>
        /// Port from tensorflow\c\c_api_test.cc
        /// </summary>
        [TestMethod]
        public void SetShape()
        {
            var s = new Status();
            var graph = tf.get_default_graph();

            var desc = c_api.TF_NewOperation(graph, "Placeholder", "");
            c_api.TF_SetAttrType(desc, "dtype", TF_DataType.TF_FLOAT);
            //if (!dims.empty())
            {
                //TF_SetAttrShape(desc, "shape", dims.data(), dims.size());
            }
            var op = c_api.TF_FinishOperation(desc, s);

            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            Assert.IsNotNull(op);

            // Fetch the shape, it should be completely unknown.
            var feed_out_0 = new TF_Output { oper = op, index = 0 };
            int num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s);

            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            Assert.AreEqual(-1, num_dims);

            // Set the shape to be unknown, expect no change.
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, new int[0], -1, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s);
            Assert.AreEqual(-1, num_dims);

            // Set the shape to be 2 x Unknown
            var dims = new int[] { 2, -1 };
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, dims.Length, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s);
            Assert.AreEqual(2, num_dims);

            // Get the dimension vector appropriately.
            var returned_dims = new int[dims.Length];
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            Assert.IsTrue(Enumerable.SequenceEqual(dims, returned_dims));

            // Set to a new valid shape: [2, 3]
            dims[1] = 3;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, dims.Length, s);
            //Assert.IsTrue(s.Code == TF_Code.TF_OK);

            // Fetch and see that the new value is returned.
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
            //Assert.IsTrue(s.Code == TF_Code.TF_OK);
            //Assert.IsTrue(Enumerable.SequenceEqual(dims, returned_dims));

            // Test for a scalar.
            var three = c_test_util.ScalarConst(3, graph, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            var three_out_0 = new TF_Output { oper = three.Handle };
            num_dims = c_api.TF_GraphGetTensorNumDims(graph, three_out_0, s);
            Assert.AreEqual(0, num_dims);
        }
    }
}
