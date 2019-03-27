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
    public class TensorTest : CApiTest
    {
        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, AllocateTensor)`
        /// </summary>
        [TestMethod]
        public void AllocateTensor()
        {
            /*ulong num_bytes = 6 * sizeof(float);
            long[] dims = { 2, 3 };
            Tensor t = c_api.TF_AllocateTensor(TF_DataType.TF_FLOAT, dims, 2, num_bytes);
            EXPECT_EQ(TF_DataType.TF_FLOAT, t.dtype);
            EXPECT_EQ(2, t.NDims);
            Assert.IsTrue(Enumerable.SequenceEqual(dims, t.shape));
            EXPECT_EQ(num_bytes, t.bytesize);
            t.Dispose();*/
        }

        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, MaybeMove)`
        /// </summary>
        [TestMethod]
        public void MaybeMove()
        {
            NDArray nd = np.array(2, 3);
            Tensor t = new Tensor(nd);
            Tensor o = t.MaybeMove();
            ASSERT_TRUE(o == IntPtr.Zero);  // It is unsafe to move memory TF might not own.
            t.Dispose();
        }

        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, Tensor)`
        /// </summary>
        [TestMethod]
        public void Tensor()
        {
            var nd = np.array(1f, 2f, 3f, 4f, 5f, 6f).reshape(2, 3);

            var tensor = new Tensor(nd);
            var array = tensor.Data<float>();

            EXPECT_EQ(tensor.dtype, TF_DataType.TF_FLOAT);
            EXPECT_EQ(tensor.rank, nd.ndim);
            EXPECT_EQ((int)tensor.shape[0], nd.shape[0]);
            EXPECT_EQ((int)tensor.shape[1], nd.shape[1]);
            EXPECT_EQ(tensor.bytesize, (ulong)nd.size * sizeof(float));
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<float>(), new float[] { 1, 2, 3, 4, 5, 6 }));
        }

        /// <summary>
        /// Port from tensorflow\c\c_api_test.cc
        /// `TEST(CAPI, SetShape)`
        /// </summary>
        [TestMethod]
        public void SetShape()
        {
            var s = new Status();
            var graph = new Graph();

            var feed = c_test_util.Placeholder(graph, s);
            var feed_out_0 = new TF_Output(feed, 0);

            // Fetch the shape, it should be completely unknown.
            int num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s);

            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(-1, num_dims);

            // Set the shape to be unknown, expect no change.
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, null, -1, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s);
            EXPECT_EQ(-1, num_dims);

            // Set the shape to be 2 x Unknown
            long[] dims = { 2, -1 };
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, dims.Length, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s);
            EXPECT_EQ(2, num_dims);

            // Get the dimension vector appropriately.
            var returned_dims = new long[dims.Length];
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            Assert.IsTrue(Enumerable.SequenceEqual(dims, returned_dims));

            // Set to a new valid shape: [2, 3]
            dims[1] = 3;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, dims.Length, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);

            // Fetch and see that the new value is returned.
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            Assert.IsTrue(Enumerable.SequenceEqual(dims, returned_dims));

            // Try to set 'unknown' with unknown rank on the shape and see that
            // it doesn't change.
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, null, -1, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(2, num_dims);
            EXPECT_EQ(2, (int)returned_dims[0]);
            EXPECT_EQ(3, (int)returned_dims[1]);

            // Try to set 'unknown' with same rank on the shape and see that
            // it doesn't change.
            dims[0] = -1;
            dims[1] = -1;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(2, num_dims);
            EXPECT_EQ(2, (int)returned_dims[0]);
            EXPECT_EQ(3, (int)returned_dims[1]);

            // Try to fetch a shape with the wrong num_dims
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, 5, s);
            Assert.IsTrue(s.Code == TF_Code.TF_INVALID_ARGUMENT);

            // Try to set an invalid shape (cannot change 2x3 to a 2x5).
            dims[1] = 5;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s);
            Assert.IsTrue(s.Code == TF_Code.TF_INVALID_ARGUMENT);

            // Test for a scalar.
            var three = c_test_util.ScalarConst(3, graph, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            var three_out_0 = new TF_Output(three, 0);

            num_dims = c_api.TF_GraphGetTensorNumDims(graph, three_out_0, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(0, num_dims);
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, null, num_dims, s);
            //Assert.IsTrue(s.Code == TF_Code.TF_OK);

            // graph.Dispose();
            s.Dispose();
        }
    }
}
