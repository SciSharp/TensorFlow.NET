using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp;
using Tensorflow;
using Buffer = Tensorflow.Buffer;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class OperationsTest
    {
        /// <summary>
        /// Port from tensorflow\c\c_api_test.cc
        /// `TEST(CAPI, GetAllOpList)`
        /// </summary>
        [TestMethod]
        public void GetAllOpList()
        {
            var handle = c_api.TF_GetAllOpList();
            var buffer = new Buffer(handle);
            var op_list = OpList.Parser.ParseFrom(buffer);

            var _registered_ops = new Dictionary<string, OpDef>();
            foreach (var op_def in op_list.Op)
                _registered_ops[op_def.Name] = op_def;

            // r1.14 added NN op
            var op = _registered_ops.FirstOrDefault(x => x.Key == "NearestNeighbors");
            Assert.IsTrue(op_list.Op.Count > 1000);
        }

        [TestMethod]
        public void addInPlaceholder()
        {
            var a = tf.placeholder(tf.float32);
            var b = tf.placeholder(tf.float32);
            var c = tf.add(a, b);

            using(var sess = tf.Session())
            {
                var o = sess.run(c, 
                    new FeedItem(a, 3.0f),
                    new FeedItem(b, 2.0f));
                Assert.AreEqual((float)o, 5.0f);
            }
        }

        [TestMethod]
        public void addInConstant()
        {
            var a = tf.constant(4.0f);
            var b = tf.constant(5.0f);
            var c = tf.add(a, b);

            using (var sess = tf.Session())
            {
                var o = sess.run(c);
                Assert.AreEqual((float)o, 9.0f);
            }
        }

        [TestMethod]
        public void addOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int firstIntVal = 2;
            const int secondIntVal = 3;

            var firstIntFeed = Enumerable.Repeat(firstIntVal, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(secondIntVal, rows * cols).ToArray();
            var intResult = firstIntFeed.Sum() + secondIntFeed.Sum();

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(tf.add(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator +(Tensor x, Tensor y)`
            c = tf.reduce_sum(tf.reduce_sum(a + b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator +(Tensor x, int y)`
            c = tf.reduce_sum(tf.reduce_sum(a + secondIntVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator +(int x, Tensor y)`
            c = tf.reduce_sum(tf.reduce_sum(secondIntVal + a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }
            #endregion

            #region floatTest
            const float firstFloatVal = 2.0f;
            const float secondFloatVal = 3.0f;

            var firstFloatFeed = Enumerable.Repeat(firstFloatVal, rows * cols).ToArray();
            var secondFloatFeed = Enumerable.Repeat(secondFloatVal, rows * cols).ToArray();
            var floatResult = firstFloatFeed.Sum() + secondFloatFeed.Sum();

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.add(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator +(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a + b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator +(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(a + secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator +(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(secondFloatVal + a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }
            #endregion

            #region doubleTest
            const double firstDoubleVal = 2.0;
            const double secondDoubleVal = 3.0;

            var firstDoubleFeed = Enumerable.Repeat(firstDoubleVal, rows * cols).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(secondDoubleVal, rows * cols).ToArray();
            var doubleResult = firstDoubleFeed.Sum() + secondDoubleFeed.Sum();

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.add(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator +(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a + b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator +(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(a + secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator +(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(secondFloatVal + a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }
            #endregion
        }

        [TestMethod]
        public void subOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int firstIntVal = -2;
            const int secondIntVal = 3;

            var firstIntFeed = Enumerable.Repeat(firstIntVal, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(secondIntVal, rows * cols).ToArray();
            var intResult = firstIntFeed.Sum() - secondIntFeed.Sum();
            var intResultTwo = -firstIntFeed.Sum();

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(tf.sub(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator -(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a - b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator -(Tensor x, int y)
            c = tf.reduce_sum(tf.reduce_sum(a - secondIntVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator -(int x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(secondIntVal - a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, Math.Abs(intResult));
            }

            // Testing `operator -(Tensor x)
            c = tf.reduce_sum(tf.reduce_sum(-a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResultTwo);
            }
            #endregion

            #region floatTest
            const float firstFloatVal = -2.0f;
            const float secondFloatVal = 3.0f;

            var firstFloatFeed = Enumerable.Repeat(firstFloatVal, rows * cols).ToArray();
            var secondFloatFeed = Enumerable.Repeat(secondFloatVal, rows * cols).ToArray();
            var floatResult = firstFloatFeed.Sum() - secondFloatFeed.Sum();
            var floatResultTwo = -firstFloatFeed.Sum();

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.sub(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator -(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a - b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator -(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(a - secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator -(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(secondFloatVal - a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, Math.Abs(floatResult));
            }

            // Testing `operator -(Tensor x)
            c = tf.reduce_sum(tf.reduce_sum(-a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResultTwo);
            }
            #endregion

            #region doubleTest
            const double firstDoubleVal = -2.0;
            const double secondDoubleVal = 3.0;

            var firstDoubleFeed = Enumerable.Repeat(firstDoubleVal, rows * cols).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(secondDoubleVal, rows * cols).ToArray();
            var doubleResult = firstDoubleFeed.Sum() - secondDoubleFeed.Sum();
            var doubleResultTwo = -firstDoubleFeed.Sum();

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.sub(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator -(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a - b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator -(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(a - secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator -(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(secondFloatVal - a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, Math.Abs(doubleResult));
            }

            // Testing `operator -(Tensor x)
            c = tf.reduce_sum(tf.reduce_sum(-a, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResultTwo);
            }
            #endregion
        }
    }
}
