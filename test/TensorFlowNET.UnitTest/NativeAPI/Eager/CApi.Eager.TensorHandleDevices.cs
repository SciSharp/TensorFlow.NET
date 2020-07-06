using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;

namespace TensorFlowNET.UnitTest.NativeAPI
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

            static SafeContextHandle NewContext(SafeStatusHandle status)
            {
                using var opts = c_api.TFE_NewContextOptions();
                return c_api.TFE_NewContext(opts, status);
            }

            using var ctx = NewContext(status);
            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            using (var hcpu = TestMatrixTensorHandle())
            {
                var device_name = TFE_TensorHandleDeviceName(hcpu, status);
                ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                ASSERT_TRUE(device_name.Contains("CPU:0"));

                var backing_device_name = TFE_TensorHandleBackingDeviceName(hcpu, status);
                ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                ASSERT_TRUE(backing_device_name.Contains("CPU:0"));

                // Disable the test if no GPU is present.
                string gpu_device_name = "";
                if (GetDeviceName(ctx, ref gpu_device_name, "GPU"))
                {
                    using var hgpu = TFE_TensorHandleCopyToDevice(hcpu, ctx, gpu_device_name, status);
                    ASSERT_TRUE(TF_GetCode(status) == TF_OK, TF_Message(status));

                    var retvals = new SafeTensorHandleHandle[1];
                    using (var shape_op = ShapeOp(ctx, hgpu))
                    {
                        TFE_OpSetDevice(shape_op, gpu_device_name, status);
                        ASSERT_TRUE(TF_GetCode(status) == TF_OK, TF_Message(status));
                        int num_retvals;
                        c_api.TFE_Execute(shape_op, retvals, out num_retvals, status);
                        ASSERT_TRUE(TF_GetCode(status) == TF_OK, TF_Message(status));

                        try
                        {
                            // .device of shape is GPU since the op is executed on GPU
                            device_name = TFE_TensorHandleDeviceName(retvals[0], status);
                            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                            ASSERT_TRUE(device_name.Contains("GPU:0"));

                            // .backing_device of shape is CPU since the tensor is backed by CPU
                            backing_device_name = TFE_TensorHandleBackingDeviceName(retvals[0], status);
                            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                            ASSERT_TRUE(backing_device_name.Contains("CPU:0"));
                        }
                        finally
                        {
                            retvals[0]?.Dispose();
                        }
                    }
                }
            }

            // not export api
            using var executor = TFE_ContextGetExecutorForThread(ctx);
            TFE_ExecutorWaitForAllPendingNodes(executor, status);
            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
        }
    }
}
