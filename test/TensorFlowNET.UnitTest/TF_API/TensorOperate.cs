using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.TF_API
{
    [TestClass]
    public class TensorOperate
    {
        [TestMethod]
        public void TransposeTest()
        {
            var a = tf.constant(np.array(new[, , ,] { { { { 1, 11, 2, 22 } }, { { 3, 33, 4, 44 } } },
                { { { 5, 55, 6, 66 } }, { { 7, 77, 8, 88 } } } }));
            var b = tf.transpose(a, new[] { 3, 1, 2, 0 });
            var transpose_a = tf.constant(np.array(new[, , ,] { { { { 1, 5 } }, { { 3, 7 } } },
                { { { 11, 55 } }, { { 33, 77 } } }, { { { 2, 6 } }, { { 4, 8 } } },
                { { { 22, 66 } }, { { 44, 88 } } } }));
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 4, 2, 1, 2 }, b.shape));
            Assert.IsTrue(Enumerable.SequenceEqual(transpose_a.numpy().ToArray<int>(), b.numpy().ToArray<int>()));
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
    }
}
