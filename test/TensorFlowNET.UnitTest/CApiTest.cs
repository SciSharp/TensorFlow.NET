using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    public class CApiTest : Python
    {
        protected TF_Code TF_OK = TF_Code.TF_OK;
        protected TF_DataType TF_FLOAT = TF_DataType.TF_FLOAT;

        protected void EXPECT_TRUE(bool expected)
        {
            Assert.IsTrue(expected);
        }

        protected void EXPECT_EQ(object expected, object actual)
        {
            Assert.AreEqual(expected, actual);
        }

        protected void ASSERT_EQ(object expected, object actual)
        {
            Assert.AreEqual(expected, actual);
        }

        protected void ASSERT_TRUE(bool condition)
        {
            Assert.IsTrue(condition);
        }

        protected OperationDescription TF_NewOperation(Graph graph, string opType, string opName)
        {
            return c_api.TF_NewOperation(graph, opType, opName);
        }

        protected void TF_AddInput(OperationDescription desc, TF_Output input)
        {
            c_api.TF_AddInput(desc, input);
        }

        protected Operation TF_FinishOperation(OperationDescription desc, Status s)
        {
            return c_api.TF_FinishOperation(desc, s);
        }

        protected void TF_SetAttrTensor(OperationDescription desc, string attrName, Tensor value, Status s)
        {
            c_api.TF_SetAttrTensor(desc, attrName, value, s);
        }

        protected void TF_SetAttrType(OperationDescription desc, string attrName, TF_DataType dtype)
        {
            c_api.TF_SetAttrType(desc, attrName, dtype);
        }

        protected void TF_SetAttrBool(OperationDescription desc, string attrName, bool value)
        {
            c_api.TF_SetAttrBool(desc, attrName, value);
        }

        protected TF_Code TF_GetCode(Status s)
        {
            return s.Code;
        }
    }
}
