using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.NumPy
{
    /// <summary>
    /// https://numpy.org/doc/stable/user/basics.indexing.html
    /// </summary>
    [TestClass]
    public class ArrayIndexingTest : EagerModeTestBase
    {
        [TestMethod]
        public void int_params()
        {
            var x = np.arange(24).reshape((2, 3, 4));
            x[1, 2, 3] = 1;
            var y = x[1, 2, 3];
            Assert.AreEqual(y.shape, Shape.Scalar);
            Assert.AreEqual(y, 1);

            x[0, 0] = new[] { 3, 1, 1, 2 };
            y = x[0, 0];
            Assert.AreEqual(y.shape, 4);
            Assert.AreEqual(y, new[] { 3, 1, 1, 2 });
            
            y = x[0];
            Assert.AreEqual(y.shape, (3, 4));

            var z = np.arange(12).reshape((3, 4));
            x[1] = z;
            Assert.AreEqual(x[1], z);
        }

        [TestMethod]
        public void slice_newaxis()
        {
            var x = np.arange(20).reshape((4, 5));
            var y = x[np.newaxis, ":2"];
            Assert.AreEqual(y.shape, (1, 2, 5));
        }

        [TestMethod]
        public void slice_params()
        {
            var x = np.arange(12).reshape((3, 4));
            var y = x[new Slice(0, 1), new Slice(2)];
            Assert.AreEqual(y.shape, (1, 2));
            Assert.AreEqual(y, np.array(new[] { 2, 3 }).reshape((1, 2)));
        }

        [TestMethod]
        public void slice_string_params()
        {
            var x = np.arange(12).reshape((3, 4));
            var y = x[Slice.ParseSlices("0:1,2:")];
            Assert.AreEqual(y.shape, (1, 2));
            Assert.AreEqual(y, np.array(new[] { 2, 3 }).reshape((1, 2)));
        }

        [TestMethod]
        public void slice_out_bound()
        {
            var input_shape = tf.constant(new int[] { 1, 1 });
            var input_shape_val = input_shape.numpy();
            input_shape_val[(int)input_shape.size - 1] = 1;
            input_shape.Dispose();
        }

        [TestMethod]
        public void shape_helper_get_shape_3dim()
        {
            var x = np.arange(24).reshape((4, 3, 2));
            var shape1 = ShapeHelper.GetShape(x.shape, new Slice(1, isIndex: true));
            Assert.AreEqual(shape1, (3, 2));

            var shape2 = ShapeHelper.GetShape(x.shape, new Slice(1));
            Assert.AreEqual(shape2, (3, 3, 2));

            var shape3 = ShapeHelper.GetShape(x.shape, new Slice(2), Slice.All);
            Assert.AreEqual(shape3, (2, 3, 2));

            var shape4 = ShapeHelper.GetShape(x.shape, new Slice(1, isIndex: true), new Slice(2));
            Assert.AreEqual(shape4, (1, 2));

            var shape5 = ShapeHelper.GetShape(x.shape, new Slice(1, isIndex: true), new Slice(1));
            Assert.AreEqual(shape5, (2, 2));

            var shape6 = ShapeHelper.GetShape(x.shape, new Slice(1), new Slice(1, isIndex: true), new Slice(1));
            Assert.AreEqual(shape6, (3, 1));
        }

        [TestMethod]
        public void shape_helper_get_shape_4dim()
        {
            var x = np.arange(120).reshape((4, 3, 2, 5));
            var slices = new[] { new Slice(1, isIndex: true), new Slice(1), new Slice(0, isIndex: true), new Slice(1) };
            var shape1 = ShapeHelper.GetShape(x.shape, slices);
            Assert.AreEqual(shape1, (2, 4));

            var shape2 = ShapeHelper.GetShape(x.shape, Slice.All);
            Assert.AreEqual(shape2, (4, 3, 2, 5));

            var shape3 = ShapeHelper.GetShape(x.shape, Slice.All, new Slice(0, isIndex: true));
            Assert.AreEqual(shape3, (4, 3, 2));
        }

        [TestMethod]
        public void iterating()
        {
            var array = np.array(new[,] { { 0, 3 }, { 2, 2 }, { 3, 1 } });
            int i = 0;
            foreach(var x in array)
            {
                if (i == 0)
                    Assert.AreEqual(x, new[] { 0, 3 });
                else
                    Assert.AreEqual(x, array[i]);
                i++;
            }
        }

        [TestMethod]
        public void slice_step_setter()
        {
            var array = np.arange(32).reshape((4, 8));
            var s1 = array[Slice.All, new Slice(2, 5, 2)] + 1;
            Assert.AreEqual(s1.shape, (4, 2));
            var expected = new[] { 3, 5, 11, 13, 19, 21, 27, 29 };
            Assert.IsTrue(Enumerable.SequenceEqual(expected, s1.ToArray<int>()));
            array[Slice.All, new Slice(2, 5, 2)] = s1;
            Assert.AreEqual(array[0], new[] { 0, 1, 3, 3, 5, 5, 6, 7 });
            Assert.AreEqual(array[1], new[] { 8, 9, 11, 11, 13, 13, 14, 15 });
            Assert.AreEqual(array[2], new[] { 16, 17, 19, 19, 21, 21, 22, 23 });
            Assert.AreEqual(array[3], new[] { 24, 25, 27, 27, 29, 29, 30, 31 });
        }

        [TestMethod]
        public void slice_step_setter_diff_shape()
        {
            var array = np.arange(32).reshape((4, 8));
            var s1 = np.array(new[] { 100, 200 });
            array[Slice.All, new Slice(2, 5, 2)] = s1;
            Assert.AreEqual(array[0], new[] { 0, 1, 100, 3, 200, 5, 6, 7 });
            Assert.AreEqual(array[1], new[] { 8, 9, 100, 11, 200, 13, 14, 15 });
            Assert.AreEqual(array[2], new[] { 16, 17, 100, 19, 200, 21, 22, 23 });
            Assert.AreEqual(array[3], new[] { 24, 25, 100, 27, 200, 29, 30, 31 });
        }

        [TestMethod]
        public void mask_2d_get_value()
        {
            var x = np.arange(25).reshape((5, 5));
            var y = np.array(new[] { true, false, true, false, true });
            var z = x[y];
            Assert.AreEqual(z.shape, (3, 5));
            Assert.AreEqual(z[0], new[] { 0, 1, 2, 3, 4 });
            Assert.AreEqual(z[1], new[] { 10, 11, 12, 13, 14 });
            Assert.AreEqual(z[2], new[] { 20, 21, 22, 23, 24 });
        }

        [TestMethod]
        public void mask_2d_set_value()
        {
            var x = np.arange(25).reshape((5, 5));
            var y = np.array(new[] {true, false, true, false, false});
            x[y] = 0;
            Assert.AreEqual(x[0], new[] { 0, 0, 0, 0, 0 });
            Assert.AreEqual(x[1], new[] { 5, 6, 7, 8, 9 });
            Assert.AreEqual(x[2], new[] { 0, 0, 0, 0, 0 });
            Assert.AreEqual(x[3], new[] { 15, 16, 17, 18, 19 });
        }
    }
}
