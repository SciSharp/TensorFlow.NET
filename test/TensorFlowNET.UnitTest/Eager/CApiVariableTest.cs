using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest.Eager
{
    /// <summary>
    /// tensorflow\c\eager\c_api_test.cc
    /// </summary>
    [TestClass]
    public class CApiVariableTest : CApiTest, IDisposable
    {
        Status status = new Status();

        [TestMethod]
        public void Variables()
        {

        }

        public void Dispose()
        {
            
        }
    }
}
