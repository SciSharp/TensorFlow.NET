using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest.Binding
{
    [TestClass]
    public class EagerTensorV2Test
    {
        [TestMethod]
        public void Creation()
        {
            var tensor = new EagerTensorV2(new float[,]
            {
                { 3.0f, 1.0f },
                { 1.0f, 2.0f }
            });
        }
    }
}
