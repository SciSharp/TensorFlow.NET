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
            int? a = 2;
            int? b = 3;
            var dims = new object[] {(int?) None, a, b};
            new TensorShape(dims).GetPrivate<Shape>("shape").Should().BeShaped(-1, 2, 3);
        }

        [TestMethod]
        public void Case2()
        {
            int? a = 2;
            int? b = 3;
            var dims = new object[] {(int?) None, a, b};
            new TensorShape(new object[] {dims}).GetPrivate<Shape>("shape").Should().BeShaped(-1, 2, 3);
        }

        [TestMethod]
        public void Case3()
        {
            int? a = 2;
            int? b = null;
            var dims = new object[] {(int?) None, a, b};
            new TensorShape(new object[] {dims}).GetPrivate<Shape>("shape").Should().BeShaped(-1, 2, -1);
        }

        [TestMethod]
        public void Case4()
        {
            TensorShape shape = (None, None);
            shape.GetPrivate<Shape>("shape").Should().BeShaped(-1, -1);
        }

        [TestMethod]
        public void Case5()
        {
            TensorShape shape = (1, None, 3);
            shape.GetPrivate<Shape>("shape").Should().BeShaped(1, -1, 3);
        }

        [TestMethod]
        public void Case6()
        {
            TensorShape shape = (None, 1, 2, 3, None);
            shape.GetPrivate<Shape>("shape").Should().BeShaped(-1, 1, 2, 3, -1);
        }
    }
}