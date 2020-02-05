using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;

namespace TensorFlowNET.UnitTest.NativeAPI
{
    public partial class CApiEagerTest
    {
        /// <summary>
        /// TEST(CAPI, TestTFE_OpInferMixedTypeInputListAttrs)
        /// </summary>
        [TestMethod]
        public unsafe void OpInferMixedTypeInputListAttrs()
        {
            using var status = TF_NewStatus();

            static SafeContextHandle NewContext(SafeStatusHandle status)
            {
                using var opts = c_api.TFE_NewContextOptions();
                return c_api.TFE_NewContext(opts, status);
            }

            using var ctx = NewContext(status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            var condition = TestScalarTensorHandle(true);
            var t1 = TestMatrixTensorHandle();
            var t2 = TestAxisTensorHandle();
            var assertOp = TFE_NewOp(ctx, "Assert", status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpAddInput(assertOp, condition, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            var data = new[] { condition, t1, t2 };
            TFE_OpAddInputList(assertOp, data, 3, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            /*var attr_values = Graph.TFE_GetOpDef("Assert").Attr;
            var attr_found = attr_values.First(x => x.Name == "T");
            EXPECT_NE(attr_found, attr_values.Last());*/
            // EXPECT_EQ(attr_found.Type[0], "DT_BOOL");
            //EXPECT_EQ(attr_found->second.list().type(1), tensorflow::DataType::DT_FLOAT);
            //EXPECT_EQ(attr_found->second.list().type(2), tensorflow::DataType::DT_INT32);

            var retvals = new IntPtr[1];
            int num_retvals = 1;
            TFE_Execute(assertOp, retvals, ref num_retvals, status);
            EXPECT_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            TFE_DeleteOp(assertOp);
            TFE_DeleteTensorHandle(condition);
            TFE_DeleteTensorHandle(t1);
            TFE_DeleteTensorHandle(t2);
            TFE_DeleteTensorHandle(retvals[0]);
        }
    }
}
