using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Lite;

namespace Tensorflow.Native.UnitTest
{
    [TestClass]
    public class TfLiteTest
    {
        [TestMethod]
        public void TfLiteVersion()
        {
            var ver = c_api_lite.StringPiece(c_api_lite.TfLiteVersion());
            Assert.IsNotNull(ver);
        }

        [TestMethod]
        public void SmokeTest()
        {
            var model = c_api_lite.TfLiteModelCreateFromFile("Lite/testdata/add.bin");
            var options = c_api_lite.TfLiteInterpreterOptionsCreate();
            c_api_lite.TfLiteInterpreterOptionsSetNumThreads(options, 2);

            var interpreter = c_api_lite.TfLiteInterpreterCreate(model, options);

            c_api_lite.TfLiteInterpreterOptionsDelete(options.DangerousGetHandle());
            c_api_lite.TfLiteModelDelete(model.DangerousGetHandle());

            Assert.AreEqual(TfLiteStatus.kTfLiteOk, c_api_lite.TfLiteInterpreterAllocateTensors(interpreter));
            Assert.AreEqual(1, c_api_lite.TfLiteInterpreterGetInputTensorCount(interpreter));
            Assert.AreEqual(1, c_api_lite.TfLiteInterpreterGetOutputTensorCount(interpreter));

            var input_dims = new int[] { 2 };
            Assert.AreEqual(TfLiteStatus.kTfLiteOk, c_api_lite.TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, input_dims.Length));
            Assert.AreEqual(TfLiteStatus.kTfLiteOk, c_api_lite.TfLiteInterpreterAllocateTensors(interpreter));

            var input_tensor = c_api_lite.TfLiteInterpreterGetInputTensor(interpreter, 0);
            Assert.AreEqual(TF_DataType.TF_FLOAT, c_api_lite.TfLiteTensorType(input_tensor));
            Assert.AreEqual(1, c_api_lite.TfLiteTensorNumDims(input_tensor));
            Assert.AreEqual(2, c_api_lite.TfLiteTensorDim(input_tensor, 0));
            Assert.AreEqual(sizeof(float) * 2, c_api_lite.TfLiteTensorByteSize(input_tensor));
            Assert.IsNotNull(c_api_lite.TfLiteTensorData(input_tensor));
            Assert.AreEqual("input", c_api_lite.StringPiece(c_api_lite.TfLiteTensorName(input_tensor)));

            var input_params = c_api_lite.TfLiteTensorQuantizationParams(input_tensor);
            Assert.AreEqual(0f, input_params.scale);
            Assert.AreEqual(0, input_params.zero_point);

            var input = new[] { 1f, 3f };
            // c_api_lite.TfLiteTensorCopyFromBuffer(input_tensor, input, 2 * sizeof(float));
        }
    }
}
