using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
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
        public void isFinite()
        {
            var a = tf.constant(new[] { 1, np.nan, 2, np.nan, 3, np.nan, 4, np.nan });
            var b = tf.cast(tf.is_finite(a), tf.float32);
            var check = np.array(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);

            using (var sess = tf.Session())
            {
                var o = sess.run(b);
                Assert.IsTrue(o.array_equal(check));
            }
        }

        [TestMethod]
        public void isNan()
        {
            var a = tf.constant(new[] { 1, np.nan, 2, np.nan, 3, np.nan, 4, np.nan });
            var b = tf.cast(tf.is_nan(a), tf.float32);
            var check = np.array(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

            using (var sess = tf.Session())
            {
                var o = sess.run(b);
                Assert.IsTrue(o.array_equal(check));
            }
        }

        [TestMethod]
        public void cumSumTest()
        {
            var a = tf.constant(new[] { 1, 1, 2, 3, 4, 5 });
            var b = tf.cumsum(a);
            var check = np.array(1, 2, 4, 7, 11, 16);

            using (var sess = tf.Session())
            {
                var o = sess.run(b);
                Assert.IsTrue(o.array_equal(check));
            }

            b = tf.cumsum(a, exclusive: true);
            check = np.array(0, 1, 2, 4, 7, 11);

            using (var sess = tf.Session())
            {
                var o = sess.run(b);
                Assert.IsTrue(o.array_equal(check));
            }

            b = tf.cumsum(a, reverse: true);
            check = np.array(16, 15, 14, 12, 9, 5);

            using (var sess = tf.Session())
            {
                var o = sess.run(b);
                Assert.IsTrue(o.array_equal(check));
            }

            b = tf.cumsum(a, exclusive:true, reverse: true);
            check = np.array(15, 14, 12, 9, 5, 0);

            using (var sess = tf.Session())
            {
                var o = sess.run(b);
                Assert.IsTrue(o.array_equal(check));
            }
        }

        [TestMethod]
        public void logicalAndTest()
        {
            var a = tf.constant(new[] {1f, 2f, 3f, 4f, -4f, -3f, -2f, -1f});
            var b = tf.less(a, 0f);
            var c = tf.greater(a, 0f);
            var d = tf.cast(tf.logical_and(b, c), tf.int32);
            var check = np.array(new[] { 0, 0, 0, 0, 0, 0, 0, 0 });

            using (var sess = tf.Session())
            {
                var o = sess.run(d);
                Assert.IsTrue(o.array_equal(check));
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

        private IEnumerable<int> MultiplyArray(IReadOnlyCollection<int> first, IReadOnlyCollection<int> second)
        {
            if(first.Count != second.Count)
                throw new ArgumentException("Arrays should be of equal size!");

            var firstEnumerator = first.GetEnumerator();
            var secondEnumerator = second.GetEnumerator();
            var result = new List<int>();
            while (firstEnumerator.MoveNext())
            {
                secondEnumerator.MoveNext();
                result.Add(firstEnumerator.Current * secondEnumerator.Current);
            }

            firstEnumerator.Dispose();
            secondEnumerator.Dispose();

            return result;
        }
        private IEnumerable<float> MultiplyArray(IReadOnlyCollection<float> first, IReadOnlyCollection<float> second)
        {
            if(first.Count != second.Count)
                throw new ArgumentException("Arrays should be of equal size!");

            var firstEnumerator = first.GetEnumerator();
            var secondEnumerator = second.GetEnumerator();
            var result = new List<float>();
            while (firstEnumerator.MoveNext())
            {
                secondEnumerator.MoveNext();
                result.Add(firstEnumerator.Current * secondEnumerator.Current);
            }

            firstEnumerator.Dispose();
            secondEnumerator.Dispose();

            return result;
        }
        private IEnumerable<double> MultiplyArray(IReadOnlyCollection<double> first, IReadOnlyCollection<double> second)
        {
            if(first.Count != second.Count)
                throw new ArgumentException("Arrays should be of equal size!");

            var firstEnumerator = first.GetEnumerator();
            var secondEnumerator = second.GetEnumerator();
            var result = new List<double>();
            while (firstEnumerator.MoveNext())
            {
                secondEnumerator.MoveNext();
                result.Add(firstEnumerator.Current * secondEnumerator.Current);
            }

            firstEnumerator.Dispose();
            secondEnumerator.Dispose();

            return result;
        }

        [TestMethod]
        public void mulOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int firstIntVal = 2;
            const int secondIntVal = 3;

            var firstIntFeed = Enumerable.Repeat(firstIntVal, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(secondIntVal, rows * cols).ToArray();
            var intResult = MultiplyArray(firstIntFeed, secondIntFeed).Sum();

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(tf.multiply(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator *(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a * b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator *(Tensor x, int y)
            c = tf.reduce_sum(tf.reduce_sum(a * secondIntVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator *(int x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(firstIntVal * b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }
            #endregion

            #region floatTest
            const float firstFloatVal = 2.0f;
            const float secondFloatVal = 3.0f;

            var firstFloatFeed = Enumerable.Repeat(firstFloatVal, rows * cols).ToArray();
            var secondFloatFeed = Enumerable.Repeat(secondFloatVal, rows * cols).ToArray();
            var floatResult = MultiplyArray(firstFloatFeed, secondFloatFeed).Sum();

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.multiply(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator *(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a * b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator *(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(a * secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator *(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(firstFloatVal * b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }
            #endregion

            #region doubleTest
            const double firstDoubleVal = 2.0;
            const double secondDoubleVal = 3.0;

            var firstDoubleFeed = Enumerable.Repeat(firstDoubleVal, rows * cols).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(secondDoubleVal, rows * cols).ToArray();
            var doubleResult = MultiplyArray(firstDoubleFeed, secondDoubleFeed).Sum();

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.multiply(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator *(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a * b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator *(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(a * secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator *(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(firstFloatVal * b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double) o, doubleResult);
            }
            #endregion
        }

        [TestMethod]
        public void divOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int firstIntVal = 6;
            const int secondIntVal = 3;

            var firstIntFeed = Enumerable.Repeat(firstIntVal, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(secondIntVal, rows * cols).ToArray();
            var intResult = (int)(firstIntFeed.Sum() / (float)secondIntVal);

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(gen_math_ops.floor_div(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator /(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a / b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator /(Tensor x, int y)
            c = tf.reduce_sum(tf.reduce_sum(a / secondIntVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator /(int x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(firstIntVal / b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }
            #endregion

            #region floatTest
            const float firstFloatVal = 6.0f;
            const float secondFloatVal = 3.0f;

            var firstFloatFeed = Enumerable.Repeat(firstFloatVal, rows * cols).ToArray();
            var secondFloatFeed = Enumerable.Repeat(secondFloatVal, rows * cols).ToArray();
            var floatResult = MultiplyArray(firstFloatFeed, secondFloatFeed.Select(x => 1/x).ToArray()).Sum();

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.divide(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator /(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a / b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator /(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(a / secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }

            // Testing `operator /(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(firstFloatVal / b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((float)o, floatResult);
            }
            #endregion

            #region doubleTest
            const double firstDoubleVal = 6.0;
            const double secondDoubleVal = 3.0;

            var firstDoubleFeed = Enumerable.Repeat(firstDoubleVal, rows * cols).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(secondDoubleVal, rows * cols).ToArray();
            var doubleResult = MultiplyArray(firstDoubleFeed, secondDoubleFeed.Select(x => 1/x).ToArray()).Sum();

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.divide(a, b), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator /(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(a / b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator /(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(a / secondFloatVal, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }

            // Testing `operator /(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(firstFloatVal / b, 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((double)o, doubleResult);
            }
            #endregion
        }

        [TestMethod]
        public void greaterThanOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int intThreshold = 10;

            var firstIntFeed = Enumerable.Range(0, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(intThreshold, rows * cols).ToArray();
            var intResult = firstIntFeed.Count(elem => elem > intThreshold);
            var intResultTwo = firstIntFeed.Count(elem => elem < intThreshold);

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator >(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a > b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator >(Tensor x, int y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a > intThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator >(int x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(intThreshold > a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResultTwo);
            }
            #endregion

            #region floatTest
            const float floatThreshold = 10.0f;

            var firstFloatFeed = Enumerable.Range(0, rows * cols).Select(elem => (float)elem).ToArray();
            var secondFloatFeed = Enumerable.Repeat(floatThreshold, rows * cols).ToArray();
            var floatResult = firstFloatFeed.Count(elem => elem > floatThreshold);
            var floatResultTwo = firstFloatFeed.Count(elem => elem < floatThreshold);

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator >(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a > b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator >(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a > floatThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator >(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(floatThreshold > a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResultTwo);
            }
            #endregion

            #region doubleTest
            const double doubleThreshold = 10.0;

            var firstDoubleFeed = Enumerable.Repeat(0, rows * cols).Select(elem => (double)elem).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(doubleThreshold, rows * cols).ToArray();
            var doubleResult = firstDoubleFeed.Count(elem => elem > doubleThreshold);
            var doubleResultTwo = firstDoubleFeed.Count(elem => elem < doubleThreshold);

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator >(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a > b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator >(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a > doubleThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator >(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(doubleThreshold > a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResultTwo);
            }
            #endregion
        }

        [TestMethod]
        public void lessThanOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int intThreshold = 10;

            var firstIntFeed = Enumerable.Range(0, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(intThreshold, rows * cols).ToArray();
            var intResult = firstIntFeed.Count(elem => elem < intThreshold);
            var intResultTwo = firstIntFeed.Count(elem => elem > intThreshold);

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.less(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator <(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a < b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator <(Tensor x, int y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a < intThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator <(int x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(intThreshold < a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResultTwo);
            }
            #endregion

            #region floatTest
            const float floatThreshold = 10.0f;

            var firstFloatFeed = Enumerable.Range(0, rows * cols).Select(elem => (float)elem).ToArray();
            var secondFloatFeed = Enumerable.Repeat(floatThreshold, rows * cols).ToArray();
            var floatResult = firstFloatFeed.Count(elem => elem < floatThreshold);
            var floatResultTwo = firstFloatFeed.Count(elem => elem > floatThreshold);

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.less(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator <(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a < b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator <(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a < floatThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator <(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(floatThreshold < a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResultTwo);
            }
            #endregion

            #region doubleTest
            const double doubleThreshold = 10.0;

            var firstDoubleFeed = Enumerable.Repeat(0, rows * cols).Select(elem => (double)elem).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(doubleThreshold, rows * cols).ToArray();
            var doubleResult = firstDoubleFeed.Count(elem => elem < doubleThreshold);
            var doubleResultTwo = firstDoubleFeed.Count(elem => elem > doubleThreshold);

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.less(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator <(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a < b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator <(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a < doubleThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator <(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(doubleThreshold < a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResultTwo);
            }
            #endregion
        }

        [TestMethod]
        public void greaterOrEqualThanOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int intThreshold = 10;

            var firstIntFeed = Enumerable.Range(0, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(intThreshold, rows * cols).ToArray();
            var intResult = firstIntFeed.Count(elem => elem >= intThreshold);
            var intResultTwo = firstIntFeed.Count(elem => elem <= intThreshold);

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater_equal(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator >=(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a >= b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator >=(Tensor x, int y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a >= intThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator >=(int x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(intThreshold >= a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResultTwo);
            }
            #endregion

            #region floatTest
            const float floatThreshold = 10.0f;

            var firstFloatFeed = Enumerable.Range(0, rows * cols).Select(elem => (float)elem).ToArray();
            var secondFloatFeed = Enumerable.Repeat(floatThreshold, rows * cols).ToArray();
            var floatResult = firstFloatFeed.Count(elem => elem >= floatThreshold);
            var floatResultTwo = firstFloatFeed.Count(elem => elem <= floatThreshold);

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater_equal(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator >=(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a >= b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator >=(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a >= floatThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator >=(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(floatThreshold >= a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResultTwo);
            }
            #endregion

            #region doubleTest
            const double doubleThreshold = 10.0;

            var firstDoubleFeed = Enumerable.Repeat(0, rows * cols).Select(elem => (double)elem).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(doubleThreshold, rows * cols).ToArray();
            var doubleResult = firstDoubleFeed.Count(elem => elem >= doubleThreshold);
            var doubleResultTwo = firstDoubleFeed.Count(elem => elem <= doubleThreshold);

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater_equal(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator >=(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a >= b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator >=(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a >= doubleThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator >=(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(doubleThreshold >= a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResultTwo);
            }
            #endregion
        }

        [TestMethod]
        public void lessOrEqualThanOpTests()
        {
            const int rows = 2; // to avoid broadcasting effect
            const int cols = 10;

            #region intTest
            const int intThreshold = 10;

            var firstIntFeed = Enumerable.Range(0, rows * cols).ToArray();
            var secondIntFeed = Enumerable.Repeat(intThreshold, rows * cols).ToArray();
            var intResult = firstIntFeed.Count(elem => elem <= intThreshold);
            var intResultTwo = firstIntFeed.Count(elem => elem >= intThreshold);

            var a = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var b = tf.placeholder(tf.int32, shape: new TensorShape(rows, cols));
            var c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.less_equal(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator <=(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a <= b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator <=(Tensor x, int y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a <= intThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResult);
            }

            // Testing `operator <=(int x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(intThreshold <= a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstIntFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, intResultTwo);
            }
            #endregion

            #region floatTest
            const float floatThreshold = 10.0f;

            var firstFloatFeed = Enumerable.Range(0, rows * cols).Select(elem => (float)elem).ToArray();
            var secondFloatFeed = Enumerable.Repeat(floatThreshold, rows * cols).ToArray();
            var floatResult = firstFloatFeed.Count(elem => elem <= floatThreshold);
            var floatResultTwo = firstFloatFeed.Count(elem => elem >= floatThreshold);

            a = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float32, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.less_equal(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator <=(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a <= b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator <=(Tensor x, float y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a <= floatThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResult);
            }

            // Testing `operator <=(float x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(floatThreshold <= a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstFloatFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, floatResultTwo);
            }
            #endregion

            #region doubleTest
            const double doubleThreshold = 10.0;

            var firstDoubleFeed = Enumerable.Repeat(0, rows * cols).Select(elem => (double)elem).ToArray();
            var secondDoubleFeed = Enumerable.Repeat(doubleThreshold, rows * cols).ToArray();
            var doubleResult = firstDoubleFeed.Count(elem => elem <= doubleThreshold);
            var doubleResultTwo = firstDoubleFeed.Count(elem => elem >= doubleThreshold);

            a = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            b = tf.placeholder(tf.float64, shape: new TensorShape(rows, cols));
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.less_equal(a, b), tf.int32), 1));

            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator <=(Tensor x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a <= b, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))),
                    new FeedItem(b, new NDArray(secondDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator <=(Tensor x, double y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(a <= doubleThreshold, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResult);
            }

            // Testing `operator <=(double x, Tensor y)
            c = tf.reduce_sum(tf.reduce_sum(tf.cast(doubleThreshold <= a, tf.int32), 1));
            using (var sess = tf.Session())
            {
                var o = sess.run(c,
                    new FeedItem(a, new NDArray(firstDoubleFeed, new Shape(rows, cols))));
                Assert.AreEqual((int)o, doubleResultTwo);
            }
            #endregion
        }
    }
}
