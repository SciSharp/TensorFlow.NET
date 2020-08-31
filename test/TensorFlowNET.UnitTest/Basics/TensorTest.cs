﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using FluentAssertions;
using Tensorflow;
using static Tensorflow.Binding;
using Tensorflow.Framework;

namespace TensorFlowNET.UnitTest.NativeAPI
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
                using (var t = new Tensor((IntPtr) ptr, new long[] {span.Length}, tf.float32, 4 * span.Length))
                {
                    Assert.IsFalse(t.IsDisposed);
                    Assert.AreEqual(2000, (int) t.bytesize);
                }
            }

            fixed (float* ptr = &array[0])
            {
                using (var t = new Tensor((IntPtr) ptr, new long[] {array.Length}, tf.float32, 4 * array.Length))
                {
                    Assert.IsFalse(t.IsDisposed);
                    Assert.AreEqual(4000, (int) t.bytesize);
                }
            }
        }

        [TestMethod]
        public unsafe void TensorFromArray()
        {
            var array = new float[1000];
            using (var t = new Tensor(array, new long[] {array.Length}, tf.float32))
            {
                Assert.IsFalse(t.IsDisposed);
                Assert.AreEqual(1000 * sizeof(float), (int) t.bytesize);
            }

            using (var t = new Tensor(new float[] {1}, new long[] {1}, tf.float32))
            {
                Assert.IsFalse(t.IsDisposed);
                Assert.AreEqual(1 * sizeof(float), (int) t.bytesize);
            }

            using (var t = new Tensor(new float[] {1}, null, tf.float32))
            {
                Assert.IsFalse(t.IsDisposed);
                Assert.AreEqual(1 * sizeof(float), (int) t.bytesize);
                t.shape.Should().BeEmpty();
            }
        }

        [TestMethod]
        public void AllocateTensor()
        {
            ulong num_bytes = 6 * sizeof(float);
            long[] dims = {2, 3};
            Tensor t = c_api.TF_AllocateTensor(TF_DataType.TF_FLOAT, dims, 2, num_bytes);
            EXPECT_EQ(TF_DataType.TF_FLOAT, t.dtype);
            EXPECT_EQ(2, t.NDims);
            EXPECT_EQ((int) dims[0], t.shape[0]);
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
            EXPECT_EQ((int) tensor.shape[0], nd.shape[0]);
            EXPECT_EQ((int) tensor.shape[1], nd.shape[1]);
            EXPECT_EQ(tensor.bytesize, (ulong) nd.size * sizeof(float));
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<float>(), new float[] {1, 2, 3, 4, 5, 6}));
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
            long[] dims = {2, -1};
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
            EXPECT_EQ(2, (int) returned_dims[0]);
            EXPECT_EQ(3, (int) returned_dims[1]);

            // Try to set 'unknown' with same rank on the shape and see that
            // it doesn't change.
            dims[0] = -1;
            dims[1] = -1;
            c_api.TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s.Handle);
            Assert.IsTrue(s.Code == TF_Code.TF_OK);
            EXPECT_EQ(2, num_dims);
            EXPECT_EQ(2, (int) returned_dims[0]);
            EXPECT_EQ(3, (int) returned_dims[1]);

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
            c_api.TF_GraphGetTensorShape(graph, feed_out_0, null, num_dims, s.Handle);
            //Assert.IsTrue(s.Code == TF_Code.TF_OK);

            // graph.Dispose();
            s.Dispose();
        }

        [TestMethod]
        public void sparse_to_dense()
        {
            var indices = tf.reshape(tf.range(0, 5), new int[] { 5, 1 });
            var labels = tf.expand_dims(tf.constant(new[] { 0, 1, 2, 3, 4 }),1);
            var st = tf.concat(values: new[] { indices, labels }, axis: 1);
            var onehot = tf.sparse_to_dense(st, (5, 5), 1);
            using (var sess = tf.Session())
            {
                var result = sess.run(onehot);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 1, 0, 0, 0, 0 }, result[0].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 1, 0, 0, 0 }, result[1].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 1, 0, 0 }, result[2].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 1, 0 }, result[3].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0, 1 }, result[4].ToArray<int>()));
            };
        }

        [TestMethod]
        public void sparse_tensor_to_dense()
        {
            var decoded_list = tf.SparseTensor(new[,]
            {
                { 0L, 0L },
                { 1L, 2L }
            },
            new int[] { 1, 2 },
            new[] { 3L, 4L });

            var onehot = tf.sparse_tensor_to_dense(decoded_list);
            using (var sess = tf.Session())
            {
                var result = sess.run(onehot);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 1, 0, 0, 0 }, result[0].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 2, 0 }, result[1].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0 }, result[2].ToArray<int>()));
            }
        }

        [TestMethod]
        public void batch_to_space_nd()
        {
            var inputs = np.arange(24).reshape(4, 2, 3);
            var block_shape = new[] { 2, 2 };
            int[,] crops = { { 0, 0 }, { 0, 0 } };
            var tensor = tf.batch_to_space_nd(inputs, block_shape, crops);

            using (var sess = tf.Session())
            {
                var result = sess.run(tensor);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 6, 1, 7, 2, 8 }, result[0, 0].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 12, 18, 13, 19, 14, 20 }, result[0, 1].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 9, 4, 10, 5, 11 }, result[0, 2].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 15, 21, 16, 22, 17, 23 }, result[0, 3].ToArray<int>()));
            }
        }

        [TestMethod, Ignore]
        public void boolean_mask()
        {
            var tensor = new[] { 0, 1, 2, 3 };
            var mask = np.array(new[] { true, false, true, false });
            var masked = tf.boolean_mask(tensor, mask);
            using (var sess = tf.Session())
            {
                var result = sess.run(masked);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 2 }, masked.ToArray<int>()));
            }
        }

        /// <summary>
        /// Creates a tensor from an image of 256x256x3 and resizes it to 100x100x3
        /// </summary>
        [TestMethod]
        public unsafe void tensor_resize()
        {
            var imageArray = new float[256 * 256 * 3];

            using var newSize = tf.convert_to_tensor(new int[] { 100, 100 });

            using (var t = new Tensor(imageArray, new long[] { 1, 256, 256, 3 }, tf.float32))
            {
                Assert.IsFalse(t.IsDisposed);
                Assert.AreEqual(256 * 256 * 3 * sizeof(float), (int)t.bytesize);

                using var resized = tf.image.resize_bilinear(t, newSize);
                EXPECT_EQ((int)resized.shape[0], 1);
                EXPECT_EQ((int)resized.shape[1], 100);
                EXPECT_EQ((int)resized.shape[2], 100);
                EXPECT_EQ((int)resized.shape[3], 3);
            }

            fixed (float* ptr = &imageArray[0])
            {
                using (var t = new Tensor((IntPtr)ptr, new long[] { imageArray.Length }, tf.float32, 4 * imageArray.Length))
                {
                    Assert.IsFalse(t.IsDisposed);
                    Assert.AreEqual(256 * 256 * 3 * sizeof(float), (int)t.bytesize);

                    using var resized = tf.image.resize_bilinear(t, newSize);
                    EXPECT_EQ((int)resized.shape[0], 1);
                    EXPECT_EQ((int)resized.shape[1], 100);
                    EXPECT_EQ((int)resized.shape[2], 100);
                    EXPECT_EQ((int)resized.shape[3], 3);
                }
            }
        }
    }
}