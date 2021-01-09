using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Runtime.InteropServices;

namespace Tensorflow.Native.UnitTest
{
    /// <summary>
    /// tensorflow\c\c_api_test.cc
    /// `class CApiColocationTest`
    /// </summary>
    [TestClass]
    public class CApiColocationTest : CApiTest, IDisposable
    {
        private Graph graph_ = new Graph();
        private Status s_ = new Status();
        private Operation feed1_;
        private Operation feed2_;
        private Operation constant_;
        private OperationDescription desc_;

        [TestInitialize]
        public void SetUp()
        {
            feed1_ = c_test_util.Placeholder(graph_, s_, "feed1");
            s_.Check();
            feed2_ = c_test_util.Placeholder(graph_, s_, "feed2");
            s_.Check();
            constant_ = c_test_util.ScalarConst(10, graph_, s_);
            s_.Check();

            desc_ = graph_.NewOperation("AddN", "add");
            TF_Output[] inputs = { new TF_Output(feed1_, 0), new TF_Output(constant_, 0) };
            desc_.AddInputList(inputs);
        }

        private void SetViaStringList(OperationDescription desc, string[] list)
        {
            var list_ptrs = new IntPtr[list.Length];
            var list_lens = new uint[list.Length];
            StringVectorToArrays(list, list_ptrs, list_lens);
            c_api.TF_SetAttrStringList(desc, "_class", list_ptrs, list_lens, list.Length);
        }

        private void StringVectorToArrays(string[] v, IntPtr[] ptrs, uint[] lens)
        {
            for (int i = 0; i < v.Length; ++i)
            {
                ptrs[i] = Marshal.StringToHGlobalAnsi(v[i]);
                lens[i] = (uint)v[i].Length;
            }
        }

        private void FinishAndVerify(OperationDescription desc, string[] expected)
        {
            var op = desc_.FinishOperation(s_);
            ASSERT_EQ(TF_Code.TF_OK, s_.Code);
            VerifyCollocation(op, expected);
        }

        private void VerifyCollocation(Operation op, string[] expected)
        {
            var handle = c_api.TF_OperationGetAttrMetadata(op, "_class", s_.Handle);
            TF_AttrMetadata m = new TF_AttrMetadata();
            if (expected.Length == 0)
            {
                ASSERT_EQ(TF_Code.TF_INVALID_ARGUMENT, s_.Code);
                EXPECT_EQ("Operation 'add' has no attr named '_class'.", s_.Message);
                return;
            }
            EXPECT_EQ(TF_Code.TF_OK, s_.Code);
            // EXPECT_EQ(1, m.is_list);
            // EXPECT_EQ(expected.Length, m.list_size);
            // EXPECT_EQ(TF_AttrType.TF_ATTR_STRING, m.type);
            string[] values = new string[expected.Length];
            uint[] lens = new uint[expected.Length];
            string[] storage = new string[m.total_size];
            //c_api.TF_OperationGetAttrStringList(op, "_class", values, lens, expected.Length, storage, m.total_size, s_);
            // EXPECT_EQ(TF_Code.TF_OK, s_.Code);
            for (int i = 0; i < expected.Length; ++i)
            {
                // EXPECT_EQ(expected[i], values[i] + lens[i]);
            }
        }

        [TestMethod]
        public void ColocateWith()
        {
            c_api.TF_ColocateWith(desc_, feed1_);
            FinishAndVerify(desc_, new string[] { "loc:@feed1" });
        }

        [TestMethod]
        public void StringList()
        {
            SetViaStringList(desc_, new string[] { "loc:@feed1" });
            FinishAndVerify(desc_, new string[] { "loc:@feed1" });
        }

        public void Dispose()
        {
            graph_.Dispose();
            s_.Dispose();
        }
    }
}
