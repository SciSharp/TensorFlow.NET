using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Tensorflow.Native.UnitTest
{
    /// <summary>
    /// tensorflow\c\c_api_test.cc
    /// `class CApiAttributesTest`
    /// </summary>
    [TestClass]
    public class AttributesTestcs : CApiTest, IDisposable
    {
        private Graph graph_;
        private int counter_;
        private Status s_;

        public AttributesTestcs()
        {
            s_ = new Status();
            graph_ = new Graph();
        }

        private OperationDescription init(string type)
        {
            // Construct op_name to match the name used by REGISTER_OP in the
            // ATTR_TEST_REGISTER calls above.
            string op_name = "CApiAttributesTestOp";
            if (type.Contains("list("))
            {
                op_name += "List";
                type = type.Substring(5, type.Length - 6);
            }
            op_name += type;
            return c_api.TF_NewOperation(graph_, op_name, $"name{counter_++}");
        }

        /// <summary>
        /// REGISTER_OP for CApiAttributesTest test cases.
        /// Registers two ops, each with a single attribute called 'v'.
        /// The attribute in one op will have a type 'type', the other
        /// will have list(type). 
        /// </summary>
        /// <param name="type"></param>
        private void ATTR_TEST_REGISTER_OP(string type)
        {

        }

        private void EXPECT_TF_META(Operation oper, string attr_name, int expected_list_size, TF_AttrType expected_type, uint expected_total_size)
        {
            var m = c_api.TF_OperationGetAttrMetadata(oper, attr_name, s_.Handle);
            EXPECT_EQ(TF_Code.TF_OK, s_.Code);
            char e = expected_list_size >= 0 ? (char)1 : (char)0;
            /*EXPECT_EQ(e, m.is_list);
            EXPECT_EQ(expected_list_size, m.list_size);
            EXPECT_EQ(expected_type, m.type);
            EXPECT_EQ(expected_total_size, m.total_size);*/
        }

        [TestMethod]
        public void String()
        {
            var desc = init("string");
            c_api.TF_SetAttrString(desc, "v", "bunny", 5);

            var oper = c_api.TF_FinishOperation(desc, s_.Handle);
            //ASSERT_EQ(TF_Code.TF_OK, s_.Code);
            //EXPECT_TF_META(oper, "v", -1, TF_AttrType.TF_ATTR_STRING, 5);
            //var value = new char[5];

            //c_api.TF_OperationGetAttrString(oper, "v", value, 5, s_);
            //EXPECT_EQ(TF_Code.TF_OK, s_.Code);
            //EXPECT_EQ("bunny", value, 5));
        }

        [TestMethod]
        public void GetAttributesTest()
        {
            var desc = graph_.NewOperation("Placeholder", "node");
            desc.SetAttrType("dtype", TF_DataType.TF_FLOAT);
            long[] ref_shape = new long[3] { 1, 2, 3 };
            desc.SetAttrShape("shape", ref_shape);
            var oper = desc.FinishOperation(s_);
            var metadata = oper.GetAttributeMetadata("shape", s_);
        }

        public void Dispose()
        {
            graph_.Dispose();
            s_.Dispose();
        }
    }
}
