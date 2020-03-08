using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;
using Buffer = System.Buffer;

namespace TensorFlowNET.UnitTest.Eager
{
    public partial class CApiEagerTest
    {
        /// <summary>
        /// TEST(CAPI, TensorHandleDevices)
        /// </summary>
        [TestMethod]
        public unsafe void TensorHandleDevices()
        {
            var status = c_api.TF_NewStatus();
            var opts = TFE_NewContextOptions();
            var ctx = TFE_NewContext(opts, status);
            TFE_DeleteContextOptions(opts);
            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            var hcpu = TestMatrixTensorHandle();
            var device_name = TFE_TensorHandleDeviceName(hcpu, status);
            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            ASSERT_TRUE(device_name.Contains("CPU:0"));

            var backing_device_name = TFE_TensorHandleBackingDeviceName(hcpu, status);
            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            ASSERT_TRUE(backing_device_name.Contains("CPU:0"));

            // Disable the test if no GPU is present.
            string gpu_device_name = "";
            if(GetDeviceName(ctx, ref gpu_device_name, "GPU"))
            {
                var hgpu = TFE_TensorHandleCopyToDevice(hcpu, ctx, gpu_device_name, status);
                ASSERT_TRUE(TF_GetCode(status) == TF_OK, TF_Message(status));

                // shape_op = ShapeOp(ctx, hgpu);
            }
        }
    }
}
