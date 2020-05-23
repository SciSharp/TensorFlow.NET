using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;
using Buffer = System.Buffer;

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
            var status = TF_NewStatus();
            var opts = TFE_NewContextOptions();
            var ctx = TFE_NewContext(opts, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_DeleteContextOptions(opts);

            var input1 = TestMatrixTensorHandle();
            var input2 = TestMatrixTensorHandle();
            var identityOp = TFE_NewOp(ctx, "IdentityN", status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            // Try to retrieve lengths before building the attributes (should fail)
            EXPECT_EQ(-1, TFE_OpGetInputLength(identityOp, "input", status));
            CHECK_NE(TF_OK, TF_GetCode(status), TF_Message(status));
            EXPECT_EQ(-1, TFE_OpGetOutputLength(identityOp, "output", status));
            CHECK_NE(TF_OK, TF_GetCode(status), TF_Message(status));

            var inputs = new IntPtr[] { input1, input2 };
            TFE_OpAddInputList(identityOp, inputs, 2, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            // Try to retrieve lengths before executing the op (should work)
            EXPECT_EQ(2, TFE_OpGetInputLength(identityOp, "input", status));
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            EXPECT_EQ(2, TFE_OpGetOutputLength(identityOp, "output", status));
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            var retvals = new IntPtr[2];
            int num_retvals = 2;
            TFE_Execute(identityOp, retvals, ref num_retvals, status);
            EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            // Try to retrieve lengths after executing the op (should work)
            EXPECT_EQ(2, TFE_OpGetInputLength(identityOp, "input", status));
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            EXPECT_EQ(2, TFE_OpGetOutputLength(identityOp, "output", status));
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            TF_DeleteStatus(status);
            TFE_DeleteOp(identityOp);
            TFE_DeleteTensorHandle(input1);
            TFE_DeleteTensorHandle(input2);
            TFE_DeleteTensorHandle(retvals[0]);
            TFE_DeleteTensorHandle(retvals[1]);
            TFE_DeleteContext(ctx);
        }
    }
}
