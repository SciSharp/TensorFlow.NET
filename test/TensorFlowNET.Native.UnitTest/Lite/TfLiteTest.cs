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
        [Ignore]
        public void TfLiteVersion()
        {
            var ver = c_api_lite.StringPiece(c_api_lite.TfLiteVersion());
            Assert.IsNotNull(ver);
        }

        [TestMethod]
        [Ignore]
        public unsafe void SmokeTest()
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
            Assert.AreEqual(TfLiteDataType.kTfLiteFloat32, c_api_lite.TfLiteTensorType(input_tensor));
            Assert.AreEqual(1, c_api_lite.TfLiteTensorNumDims(input_tensor));
            Assert.AreEqual(2, c_api_lite.TfLiteTensorDim(input_tensor, 0));
            Assert.AreEqual(sizeof(float) * 2, c_api_lite.TfLiteTensorByteSize(input_tensor));
            Assert.IsNotNull(c_api_lite.TfLiteTensorData(input_tensor));
            Assert.AreEqual("input", c_api_lite.StringPiece(c_api_lite.TfLiteTensorName(input_tensor)));

            var input_params = c_api_lite.TfLiteTensorQuantizationParams(input_tensor);
            Assert.AreEqual(0f, input_params.scale);
            Assert.AreEqual(0, input_params.zero_point);

            var input = new[] { 1f, 3f };
            fixed (float* addr = &input[0])
            {
                Assert.AreEqual(TfLiteStatus.kTfLiteOk, 
                    c_api_lite.TfLiteTensorCopyFromBuffer(input_tensor, new IntPtr(addr), 2 * sizeof(float)));
            }

            Assert.AreEqual(TfLiteStatus.kTfLiteOk, c_api_lite.TfLiteInterpreterInvoke(interpreter));

            var output_tensor = c_api_lite.TfLiteInterpreterGetOutputTensor(interpreter, 0);
            Assert.AreEqual(TfLiteDataType.kTfLiteFloat32, c_api_lite.TfLiteTensorType(output_tensor));
            Assert.AreEqual(1, c_api_lite.TfLiteTensorNumDims(output_tensor));
            Assert.AreEqual(2, c_api_lite.TfLiteTensorDim(output_tensor, 0));
            Assert.AreEqual(sizeof(float) * 2, c_api_lite.TfLiteTensorByteSize(output_tensor));
            Assert.IsNotNull(c_api_lite.TfLiteTensorData(output_tensor));
            Assert.AreEqual("output", c_api_lite.StringPiece(c_api_lite.TfLiteTensorName(output_tensor)));

            var output_params = c_api_lite.TfLiteTensorQuantizationParams(output_tensor);
            Assert.AreEqual(0f, output_params.scale);
            Assert.AreEqual(0, output_params.zero_point);

            var output = new float[2];
            fixed (float* addr = &output[0])
            {
                Assert.AreEqual(TfLiteStatus.kTfLiteOk,
                    c_api_lite.TfLiteTensorCopyToBuffer(output_tensor, new IntPtr(addr), 2 * sizeof(float)));
            }
            Assert.AreEqual(3f, output[0]);
            Assert.AreEqual(9f, output[1]);

            c_api_lite.TfLiteInterpreterDelete(interpreter.DangerousGetHandle());
        }

        [TestMethod]
        [Ignore]
        public unsafe void QuantizationParamsTest()
        {
            var model = c_api_lite.TfLiteModelCreateFromFile("Lite/testdata/add_quantized.bin");
            var interpreter = c_api_lite.TfLiteInterpreterCreate(model, new SafeTfLiteInterpreterOptionsHandle(IntPtr.Zero));
            c_api_lite.TfLiteModelDelete(model.DangerousGetHandle());
            var input_dims = new[] { 2 };
            Assert.AreEqual(TfLiteStatus.kTfLiteOk, c_api_lite.TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 1));
            Assert.AreEqual(TfLiteStatus.kTfLiteOk, c_api_lite.TfLiteInterpreterAllocateTensors(interpreter));

            var input_tensor = c_api_lite.TfLiteInterpreterGetInputTensor(interpreter, 0);
            Assert.IsNotNull(input_tensor);

            Assert.AreEqual(TfLiteDataType.kTfLiteUInt8, c_api_lite.TfLiteTensorType(input_tensor));
            Assert.AreEqual(1, c_api_lite.TfLiteTensorNumDims(input_tensor));
            Assert.AreEqual(2, c_api_lite.TfLiteTensorDim(input_tensor, 0));

            var input_params = c_api_lite.TfLiteTensorQuantizationParams(input_tensor);
            Assert.AreEqual((0.003922f, 0), (input_params.scale, input_params.zero_point));

            var input = new byte[] { 1, 3 };
            fixed (byte* addr = &input[0])
            {
                Assert.AreEqual(TfLiteStatus.kTfLiteOk, 
                    c_api_lite.TfLiteTensorCopyFromBuffer(input_tensor, new IntPtr(addr), 2 * sizeof(byte)));
            }
            Assert.AreEqual(TfLiteStatus.kTfLiteOk, c_api_lite.TfLiteInterpreterInvoke(interpreter));

            var output_tensor = c_api_lite.TfLiteInterpreterGetOutputTensor(interpreter, 0);
            Assert.IsNotNull(output_tensor);

            var output_params = c_api_lite.TfLiteTensorQuantizationParams(output_tensor);
            Assert.AreEqual((0.003922f, 0), (output_params.scale, output_params.zero_point));

            var output = new byte[2];
            fixed (byte* addr = &output[0])
            {
                Assert.AreEqual(TfLiteStatus.kTfLiteOk,
                    c_api_lite.TfLiteTensorCopyToBuffer(output_tensor, new IntPtr(addr), 2 * sizeof(byte)));
            }
            Assert.AreEqual(3f, output[0]);
            Assert.AreEqual(9f, output[1]);

            var dequantizedOutput0 = output_params.scale * (output[0] - output_params.zero_point);
            var dequantizedOutput1 = output_params.scale * (output[1] - output_params.zero_point);
            Assert.AreEqual(dequantizedOutput0, 0.011766f);
            Assert.AreEqual(dequantizedOutput1, 0.035298f);

            c_api_lite.TfLiteInterpreterDelete(interpreter.DangerousGetHandle());
        }
    }
}
