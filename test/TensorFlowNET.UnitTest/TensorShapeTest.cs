using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class TensorShapeTest
    {
        [TestMethod]
        public void Case1()
        {
            int a = 2;
            int b = 3;
            var dims = new [] { Unknown, a, b};
            new TensorShape(dims).GetPrivate<Shape>("shape").Should().BeShaped(-1, 2, 3);
        }

        [TestMethod]
        public void Case2()
        {
            int a = 2;
            int b = 3;
            var dims = new[] { Unknown, a, b};
            new TensorShape(new [] {dims}).GetPrivate<Shape>("shape").Should().BeShaped(-1, 2, 3);
        }

        [TestMethod]
        public void Case3()
        {
            int a = 2;
            int b = Unknown;
            var dims = new [] { Unknown, a, b};
            new TensorShape(new [] {dims}).GetPrivate<Shape>("shape").Should().BeShaped(-1, 2, -1);
        }

        [TestMethod]
        public void Case4()
        {
            TensorShape shape = (Unknown, Unknown);
            shape.GetPrivate<Shape>("shape").Should().BeShaped(-1, -1);
        }

        [TestMethod]
        public void Case5()
        {
            TensorShape shape = (1, Unknown, 3);
            shape.GetPrivate<Shape>("shape").Should().BeShaped(1, -1, 3);
        }

        [TestMethod]
        public void Case6()
        {
            TensorShape shape = (Unknown, 1, 2, 3, Unknown);
            shape.GetPrivate<Shape>("shape").Should().BeShaped(-1, 1, 2, 3, -1);
        }

        [TestMethod]
        public void Case7()
        {
            TensorShape shape = new TensorShape();
            Assert.AreEqual(shape.rank, -1);
        }
    }
}