using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.Device;
using Tensorflow.Eager;

namespace Tensorflow.Native.UnitTest.Eager
{
    public partial class CApiEagerTest
    {
        /// <summary>
        /// TEST(CAPI, Context)
        /// </summary>
        [TestMethod]
        public void Context()
        {
            using var status = c_api.TF_NewStatus();

            static SafeContextHandle NewContext(SafeStatusHandle status)
            {
                using var opts = c_api.TFE_NewContextOptions();
                return c_api.TFE_NewContext(opts, status);
            }

            static SafeDeviceListHandle ListDevices(SafeStatusHandle status)
            {
                using var ctx = NewContext(status);
                var devices = c_api.TFE_ContextListDevices(ctx, status);
                EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                return devices;
            }

            using var devices = ListDevices(status);
            EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            int num_devices = c_api.TF_DeviceListCount(devices);
            EXPECT_GE(num_devices, 1, TF_Message(status));
            for (int i = 0; i < num_devices; ++i)
            {
                EXPECT_NE("", c_api.TF_DeviceListName(devices, i, status), TF_Message(status));
                EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            }
        }
    }
}
