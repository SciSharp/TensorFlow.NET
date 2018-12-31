using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.UnitTest
{
    public class CApiTest
    {
        public void EXPECT_EQ(object expected, object actual)
        {
            Assert.AreEqual(expected, actual);
        }

        public void ASSERT_EQ(object expected, object actual)
        {
            Assert.AreEqual(expected, actual);
        }

        public void ASSERT_TRUE(bool condition)
        {
            Assert.IsTrue(condition);
        }
    }
}
