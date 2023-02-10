using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;

namespace TensorFlowNET.UnitTest.NumPy
{
    [TestClass]
    public class OperatorsTest
    {
        [TestMethod]
        public void EqualToOperator()
        {
            NDArray n1 = null;
            NDArray n2 = new NDArray(1);

            Assert.IsTrue(n1 == null);
            Assert.IsFalse(n2 == null);
            Assert.IsFalse(n1 == 1);
            Assert.IsTrue(n2 == 1);
        }

        [TestMethod]
        public void NotEqualToOperator()
        {
            NDArray n1 = null;
            NDArray n2 = new NDArray(1);

            Assert.IsFalse(n1 != null);
            Assert.IsTrue(n2 != null);
            Assert.IsTrue(n1 != 1);
            Assert.IsFalse(n2 != 1);
        }
    }
}
