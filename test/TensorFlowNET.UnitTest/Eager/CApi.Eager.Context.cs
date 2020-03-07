using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;

namespace TensorFlowNET.UnitTest.Eager
{
    public partial class CApiEagerTest
    {
        /// <summary>
        /// TEST(CAPI, Context)
        /// </summary>
        [TestMethod]
        public void Context()
        {
            var status = c_api.TF_NewStatus();
            var opts = c_api.TFE_NewContextOptions();
            var ctx = c_api.TFE_NewContext(opts, status);

            c_api.TFE_DeleteContextOptions(opts);

            var devices = c_api.TFE_ContextListDevices(ctx, status);
            EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            c_api.TFE_DeleteContext(ctx);
            EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            int num_devices = c_api.TF_DeviceListCount(devices);
            EXPECT_GE(num_devices, 1, TF_Message(status));
            for (int i = 0; i < num_devices; ++i)
            {
                EXPECT_NE("", c_api.TF_DeviceListName(devices, i, status), TF_Message(status));
                EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            }

            // c_api.TF_DeleteDeviceList(devices);
            c_api.TF_DeleteStatus(status);
        }
    }
}
