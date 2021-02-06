using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using static Tensorflow.Binding;

namespace Tensorflow.Native.UnitTest.Tensors
{
    [TestClass]
    public class TensorTest : CApiTest
    {
        [TestMethod]
        public unsafe void TensorFromFixed()
        {
            var array = new float[1000];
            var span = new Span<float>(array, 100, 500);
            fixed (float* ptr = &MemoryMarshal.GetReference(span))
            {
                using (var t = new Tensor((IntPtr)ptr, new long[] { span.Length }, tf.float32, 4 * span.Length))
                {
                    Assert.IsFalse(t.IsDisposed);
                    Assert.AreEqual(2000, (int)t.bytesize);
                }
            }

            fixed (float* ptr = &array[0])
            {
                using (var t = new Tensor((IntPtr)ptr, new long[] { array.Length }, tf.float32, 4 * array.Length))
                {
                    Assert.IsFalse(t.IsDisposed);
                    Assert.AreEqual(4000, (int)t.bytesize);
                }
            }
        }

        [TestMethod]
        public void TensorFromArray()
        {
            var array = new float[1000];
            using (var t = new Tensor(array, new long[] { array.Length }, tf.float32))
            {
                Assert.IsFalse(t.IsDisposed);
                Assert.AreEqual(1000 * sizeof(float), (int)t.bytesize);
            }

            using (var t = new Tensor(new float[] { 1 }, new long[] { 1 }, tf.float32))
            {
                Assert.IsFalse(t.IsDisposed);
                Assert.AreEqual(1 * sizeof(float), (int)t.bytesize);
            }

            using (var t = new Tensor(new float[] { 1 }, null, tf.float32))
            {
                Assert.IsFalse(t.IsDisposed);
                Assert.AreEqual(1 * sizeof(float), (int)t.bytesize);
                t.shape.Should().BeEmpty();
            }
        }

        [TestMethod]
        public void AllocateTensor()
        {
            ulong num_bytes = 6 * sizeof(float);
            long[] dims = { 2, 3 };
            Tensor t = c_api.TF_AllocateTensor(TF_DataType.TF_FLOAT, dims, 2, num_bytes);
            EXPECT_EQ(TF_DataType.TF_FLOAT, t.dtype);
            EXPECT_EQ(2, t.NDims);
            EXPECT_EQ((int)dims[0], t.shape[0]);
            EXPECT_EQ(num_bytes, t.bytesize);
            t.Dispose();
        }


        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, MaybeMove)`
        /// </summary>
        [TestMethod, Ignore]
        public void MaybeMove()
        {
            NDArray nd = np.array(2, 3);
            Tensor t = new Tensor(nd);
            Tensor o = t.MaybeMove();
            ASSERT_TRUE(o == IntPtr.Zero); // It is unsafe to move memory TF might not own.
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
            var array = tensor.ToArray<float>();

            EXPECT_EQ(tensor.dtype, TF_DataType.TF_FLOAT);
            EXPECT_EQ(tensor.rank, nd.ndim);
            EXPECT_EQ((int)tensor.shape[0], nd.shape[0]);
            EXPECT_EQ((int)tensor.shape[1], nd.shape[1]);
            EXPECT_EQ(tensor.bytesize, (ulong)nd.size * sizeof(float));
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<float>(), new float[] { 1, 2, 3, 4, 5, 6 }));
        }

        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST_F(CApiAttributesTest, StringTensor)`
        /// </summary>
        [TestMethod, Ignore("Waiting for PR https://github.com/tensorflow/tensorflow/pull/46804")]
        public void StringTensor()
        {
            string text = "Hello world!.";

            var tensor = c_api.TF_AllocateTensor(TF_DataType.TF_STRING,
                null,
                0,
                1 * 24);
            var tstr = c_api.TF_StringInit(tensor);
            var data = c_api.TF_StringGetDataPointer(tstr);
            c_api.TF_StringCopy(tstr, text, text.Length);

            Assert.AreEqual((ulong)text.Length, c_api.TF_StringGetSize(tstr));
            Assert.AreEqual(text, c_api.StringPiece(data));
            Assert.AreEqual((ulong)text.Length, c_api.TF_TensorByteSize(tensor));
            Assert.AreEqual(0, c_api.TF_NumDims(tensor));

            TF_DeleteTensor(tensor);
            c_api.TF_StringDealloc(tstr);
        }

        /// <summary>
        /// Port from tensorflow\c\c_api_test.cc
        /// `TEST(CAPI, SetShape)`
        /// </summary>
        [TestMethod]
        public void SetShape()
        {
            var s = new Status();
            var graph = new Graph().as_default();

            var feed = c_test_util.Placeholder(graph, s);
            var feed_out_0 = new TF_Output(feed, 0);

            // Fetch the shape, it should be completely unknown.
            int num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s.Handle);

            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(-1, num_dims);

            // Set the shape to be unknown, expect no change.
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, null, -1, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s.Handle);
            EXPECT_EQ(-1, num_dims);

            // Set the shape to be 2 x Unknown
            long[] dims = { 2, -1 };
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, dims.Length, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            num_dims = c_api.TF_GraphGetTensorNumDims(graph, feed_out_0, s.Handle);
            EXPECT_EQ(2, num_dims);

            // Get the dimension vector appropriately.
            var returned_dims = new long[dims.Length];
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            Assert.IsTrue(Enumerable.SequenceEqual(dims, returned_dims));

            // Set to a new valid shape: [2, 3]
            dims[1] = 3;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, dims.Length, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);

            // Fetch and see that the new value is returned.
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            Assert.IsTrue(Enumerable.SequenceEqual(dims, returned_dims));

            // Try to set 'unknown' with unknown rank on the shape and see that
            // it doesn't change.
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, null, -1, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(2, num_dims);
            EXPECT_EQ(2, (int)returned_dims[0]);
            EXPECT_EQ(3, (int)returned_dims[1]);

            // Try to set 'unknown' with same rank on the shape and see that
            // it doesn't change.
            dims[0] = -1;
            dims[1] = -1;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(2, num_dims);
            EXPECT_EQ(2, (int)returned_dims[0]);
            EXPECT_EQ(3, (int)returned_dims[1]);

            // Try to fetch a shape with the wrong num_dims
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, 5, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_INVALID_ARGUMENT);

            // Try to set an invalid shape (cannot change 2x3 to a 2x5).
            dims[1] = 5;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_INVALID_ARGUMENT);

            // Test for a scalar.
            var three = c_test_util.ScalarConst(3, graph, s);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            var three_out_0 = new TF_Output(three, 0);

            num_dims = c_api.TF_GraphGetTensorNumDims(graph, three_out_0, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(0, num_dims);
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, dims, num_dims, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_INVALID_ARGUMENT);

            graph.Exit();
            s.Dispose();
        }
    }
}