using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class StatusTest
    {
        [TestMethod]
        public void NewStatus()
        {
            var s = new Status();
            Assert.AreEqual(s.Code, TF_Code.TF_OK);
            Assert.AreEqual(s.Message, String.Empty);
        }

        [TestMethod]
        public void SetStatus()
        {
            var s = new Status();
            s.SetStatus(TF_Code.TF_CANCELLED, "cancel");
            Assert.AreEqual(s.Code, TF_Code.TF_CANCELLED);
            Assert.AreEqual(s.Message, "cancel");
        }

        [TestMethod]
        public void DeleteStatus()
        {
            var s = new Status();
        }
    }
}
