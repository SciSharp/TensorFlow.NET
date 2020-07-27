using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.NativeAPI
{
    public partial class CApiEagerTest
    {
        /// <summary>
        /// TEST(CAPI, Execute_MatMul_CPU)
        /// </summary>
        [TestMethod]
        public unsafe void Execute_MatMul_CPU()
        {
            Execute_MatMul_CPU(false);
        }

        unsafe void Execute_MatMul_CPU(bool async)
        {
            using var status = TF_NewStatus();

            static SafeContextHandle NewContext(bool async, SafeStatusHandle status)
            {
                using var opts = c_api.TFE_NewContextOptions();
                c_api.TFE_ContextOptionsSetAsync(opts, Convert.ToByte(async));
                return c_api.TFE_NewContext(opts, status);
            }

            IntPtr t;
            using (var ctx = NewContext(async, status))
            {
                CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

                var retvals = new SafeTensorHandleHandle[2];
                using (var m = TestMatrixTensorHandle())
                using (var matmul = MatMulOp(ctx, m, m))
                {
                    int num_retvals;
                    c_api.TFE_Execute(matmul, retvals, out num_retvals, status);
                    EXPECT_EQ(1, num_retvals);
                    EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                }

                try
                {
                    t = TFE_TensorHandleResolve(retvals[0], status);
                    ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                }
                finally
                {
                    retvals[0].Dispose();
                }
            }

            ASSERT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            var product = new float[4];
            EXPECT_EQ(product.Length * sizeof(float), (int)TF_TensorByteSize(t));
            tf.memcpy(product, TF_TensorData(t), TF_TensorByteSize(t));

            c_api.TF_DeleteTensor(t);
            EXPECT_EQ(7f, product[0]);
            EXPECT_EQ(10f, product[1]);
            EXPECT_EQ(15f, product[2]);
            EXPECT_EQ(22f, product[3]);
        }
    }
}
