using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using System;
using System.Linq;
using static Tensorflow.Binding;
using Tensorflow;

namespace TensorFlowNET.UnitTest.NumPy
{
    [TestClass]
    public class ShapeTest : EagerModeTestBase
    {
        [Ignore]
        [TestMethod]
        public unsafe void ShapeGetLastElements()
        {
            // test code from function _CheckAtLeast3DImage
            // 之前的 _CheckAtLeast3DImage 有bug，现在通过测试，下面的代码是正确的
            // todo: shape["-3:"] 的写法，目前有bug，需要修复，单元测试等修复后再放开，暂时先忽略测试

            var image_shape = new Shape(new[] { 32, 64, 3 });
            var image_shape_4d = new Shape(new[] { 4, 64, 32, 3 });

            var image_shape_last_three_elements = new Shape(new[] {
                                                image_shape.dims[image_shape.dims.Length - 3],
                                                image_shape.dims[image_shape.dims.Length - 2],
                                                image_shape.dims[image_shape.dims.Length - 1]});

            var image_shape_last_three_elements2 = image_shape["-3:"];

            Assert.IsTrue(Equal(image_shape_last_three_elements.dims, image_shape_last_three_elements2.dims), "3dims get fail.");

            var image_shape_last_three_elements_4d = new Shape(new[] {
                                                image_shape_4d.dims[image_shape_4d.dims.Length - 3],
                                                image_shape_4d.dims[image_shape_4d.dims.Length - 2],
                                                image_shape_4d.dims[image_shape_4d.dims.Length - 1]});

            var image_shape_last_three_elements2_4d = image_shape_4d["-3:"];

            Assert.IsTrue(Equals(image_shape_last_three_elements_4d.dims, image_shape_last_three_elements2_4d.dims), "4dims get fail.");
        }

    }
}