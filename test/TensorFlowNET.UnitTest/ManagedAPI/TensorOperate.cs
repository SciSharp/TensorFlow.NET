using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class TensorOperate
    {
        [TestMethod]
        public void TransposeTest()
        {
            // https://www.tensorflow.org/api_docs/python/tf/transpose#for_example_2
            var x = tf.constant(new int[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            });
            var transpose_x = tf.transpose(x);
            Assert.AreEqual(new[] { 1, 4 }, transpose_x[0].numpy());
            Assert.AreEqual(new[] { 2, 5 }, transpose_x[1].numpy());
            Assert.AreEqual(new[] { 3, 6 }, transpose_x[2].numpy());

            #region constant a
            var a = tf.constant(np.array(new[, , ,]
            {
                {
                    {
                        { 1, 11, 2, 22 }
                    },
                    {
                        { 3, 33, 4, 44 }
                    }
                },
                {
                    {
                        { 5, 55, 6, 66 }
                    },
                    {
                        { 7, 77, 8, 88 }
                    }
                }
            }));

            #endregion
            var actual_transposed_a = tf.transpose(a, new[] { 3, 1, 2, 0 });

            #region constant transpose_a
            var expected_transposed_a = tf.constant(np.array(new[, , ,]
            {
                {
                    { { 1, 5 } }, { { 3, 7 } }
                },
                {
                    { { 11, 55 } }, { { 33, 77 } }
                },
                {
                    {
                        { 2, 6 }
                    },
                    {
                        { 4, 8 }
                    }
                },
                {
                    {
                        { 22, 66 }
                    },
                    {
                        { 44, 88 }
                    }
                }
            }));
            #endregion
            Assert.AreEqual((4, 2, 1, 2), actual_transposed_a.TensorShape);
            Assert.AreEqual(expected_transposed_a.numpy(), actual_transposed_a.numpy());
        }

        [TestMethod]
        public void InitTensorTest()
        {
            var a = tf.constant(np.array(new[, ,]
            {
                { { 1 }, { 2 }, { 3 } },
                { { 4 }, { 5 }, { 6 } }
            }));
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 2, 3, 1 }, a.shape));

            var b = tf.constant(new[, ,]
            {
                { { 1 }, { 2 }, { 3 } },
                { { 4 }, { 5 }, { 6 } }
            });
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 2, 3, 1 }, b.shape));
        }

        [TestMethod]
        public void ConcatTest()
        {
            var a = tf.constant(new[,] { { 1, 2 }, { 3, 4 } });
            var b = tf.constant(new[,] { { 5, 6 }, { 7, 8 } });
            var c = tf.constant(new[,] { { 9, 10 }, { 11, 12 } });

            var concatValue = tf.concat(new[] { a, b, c }, axis: 0);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 6, 2 }, concatValue.shape));
        }

        [TestMethod]
        public void ConcatDoubleTest()
        {
            var a = tf.constant(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            var b = tf.constant(new[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });
            var c = tf.constant(new[,] { { 9.0, 10.0 }, { 11.0, 12.0 } });

            var concatValue = tf.concat(new[] { a, b, c }, axis: 0);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 6, 2 }, concatValue.shape));
        }

        [TestMethod]
        public void ConcatAndSplitTest()
        {
            var a = tf.constant(new[,] { { 1, 2 }, { 3, 4 } });
            var b = tf.constant(new[,] { { 5, 6 }, { 7, 8 } });
            var c = tf.constant(new[,] { { 9, 10 }, { 11, 12 } });

            var value = tf.concat(new[] { a, b, c }, axis: 0);

            var splitValue = tf.split(value, 3, axis: 0);
            Assert.AreEqual(3, splitValue.Length);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 2, 2 }, splitValue[0].shape));
        }

        #region ones/zeros like
        [TestMethod]
        public void TestOnesLike()
        {
            #region 2-dimension
            var ones2D = tf.ones_like(new int[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            });

            Assert.AreEqual(new[] { 1, 1, 1 }, ones2D[0].numpy());
            Assert.AreEqual(new[] { 1, 1, 1 }, ones2D[1].numpy());
            #endregion

            #region 1-dimension
            var ones1D = tf.ones_like(new int[,]
            {
                { 1, 2, 3 }
            });

            Assert.AreEqual(new[] { 1, 1, 1 }, ones1D[0].numpy());
            #endregion
        }

        [TestMethod]
        public void TestZerosLike()
        {
            #region 2-dimension
            var zeros2D = tf.zeros_like(new int[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            });

            Assert.AreEqual(new[] { 0, 0, 0 }, zeros2D[0].numpy());
            Assert.AreEqual(new[] { 0, 0, 0 }, zeros2D[1].numpy());
            #endregion

            #region 1-dimension
            var zeros1D = tf.zeros_like(new int[,]
            {
                { 1, 2, 3 }
            });

            Assert.AreEqual(new[] { 0, 0, 0 }, zeros1D[0].numpy());
            #endregion
        }
        #endregion
    }
}
