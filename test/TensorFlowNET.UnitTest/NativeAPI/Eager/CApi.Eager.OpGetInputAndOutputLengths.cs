using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;

namespace TensorFlowNET.UnitTest.NativeAPI
{
    public partial class CApiEagerTest
    {
        /// <summary>
        /// TEST(CAPI, TestTFE_OpGetInputAndOutputLengths)
        /// </summary>
        [TestMethod]
        public unsafe void OpGetInputAndOutputLengths()
        {
            using var status = TF_NewStatus();

            static SafeContextHandle NewContext(SafeStatusHandle status)
            {
                using var opts = c_api.TFE_NewContextOptions();
                return c_api.TFE_NewContext(opts, status);
            }

            using var ctx = NewContext(status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            using var input1 = TestMatrixTensorHandle();
            using var input2 = TestMatrixTensorHandle();

            var retvals = new SafeTensorHandleHandle[2];
            using (var identityOp = TFE_NewOp(ctx, "IdentityN", status))
            {
                CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

                // Try to retrieve lengths before building the attributes (should fail)
                EXPECT_EQ(-1, TFE_OpGetInputLength(identityOp, "input", status));
                CHECK_NE(TF_OK, TF_GetCode(status), TF_Message(status));
                EXPECT_EQ(-1, TFE_OpGetOutputLength(identityOp, "output", status));
                CHECK_NE(TF_OK, TF_GetCode(status), TF_Message(status));

                var inputs = new SafeTensorHandleHandle[] { input1, input2 };
                TFE_OpAddInputList(identityOp, inputs, 2, status);
                CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

                // Try to retrieve lengths before executing the op (should work)
                EXPECT_EQ(2, TFE_OpGetInputLength(identityOp, "input", status));
                CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                EXPECT_EQ(2, TFE_OpGetOutputLength(identityOp, "output", status));
                CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

                int num_retvals;
                TFE_Execute(identityOp, retvals, out num_retvals, status);
                EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                EXPECT_EQ(2, num_retvals);

                try
                {
                    // Try to retrieve lengths after executing the op (should work)
                    EXPECT_EQ(2, TFE_OpGetInputLength(identityOp, "input", status));
                    CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                    EXPECT_EQ(2, TFE_OpGetOutputLength(identityOp, "output", status));
                    CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
                }
                finally
                {
                    retvals[0].Dispose();
                    retvals[1].Dispose();
                }
            }
        }
    }
}
