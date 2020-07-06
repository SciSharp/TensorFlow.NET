using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.NativeAPI
{
    public partial class CApiEagerTest
    {
        /// <summary>
        /// TEST(CAPI, TensorHandle)
        /// </summary>
        [TestMethod]
        public unsafe void TensorHandle()
        {
            using var h = TestMatrixTensorHandle();
            EXPECT_EQ(TF_FLOAT, c_api.TFE_TensorHandleDataType(h));

            var status = c_api.TF_NewStatus();
            var t = c_api.TFE_TensorHandleResolve(h, status);
            ASSERT_EQ(16ul, c_api.TF_TensorByteSize(t));

            var data = new float[] { 0f, 0f, 0f, 0f };
            tf.memcpy(data, c_api.TF_TensorData(t), data.Length * sizeof(float));

            EXPECT_EQ(1.0f, data[0]);
            EXPECT_EQ(2.0f, data[1]);
            EXPECT_EQ(3.0f, data[2]);
            EXPECT_EQ(4.0f, data[3]);
            c_api.TF_DeleteTensor(t);
        }
    }
}
