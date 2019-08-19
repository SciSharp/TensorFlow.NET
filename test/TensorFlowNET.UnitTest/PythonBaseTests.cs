using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class PythonBaseTests : PythonTest
    {
        [Ignore]
        [TestMethod]
        public void weakKeyDictionary_test()
        {
            var weakKeyDict = new WeakKeyDictionary<int, char>();
            for (int i = 0; i < 5; i++)
            {
                var c = (char)((int)'a' + i);
                weakKeyDict[i] = c;
                //Assert.AreEqual(weakKeyDict.Count, (int)(i + 1));
                var v = (weakKeyDict.Count == i + 1);
                Assert.IsTrue(v);
            }
            //Assert.AreEqual(weakKeyDict.Count, 0);
            var b = (weakKeyDict.Count == 0);
            Assert.IsTrue(b);
        }

        [TestMethod]
        public void isinstance_test()
        {
            var s1 = "hi";
            var s2 = "hello";

            var t1 = (s1, s2);
            var t2 = (s1, s2, s1);
            var t3 = (s2, s1);

            var true1 = isinstance(s1, typeof(string));
            var false1 = isinstance(t1, typeof(string));
            var true2 = isinstance(t1, t3.GetType());
            var false2 = isinstance(t1, t2.GetType());
            var true3 = isinstance(t1, (t2.GetType(), t1.GetType(), typeof(string)));
            var false3 = isinstance(t3, (t2.GetType(), typeof(string)));

            Assert.IsTrue(true1);
            Assert.IsTrue(true2);
            Assert.IsTrue(true3);
            Assert.IsFalse(false1);
            Assert.IsFalse(false2);
            Assert.IsFalse(false3);
        }

        [TestMethod]
        public void hasattr_getattr()
        {
            var s1 = "Tensorflow v0.1";
            var f = "Tensorflow";
            var r = "Tensorflow.NET";
            var res = s1.Replace(f, r);

            // Test 1
            Assert.IsTrue(hasattr(s1, "Replace"));

            // Test 2
            var o = getattr( s1, "Replace", typeof(string), typeof(string));
            Assert.AreEqual(res, o(f, r));

            // Test 3
            var l = getattr(s1, "Length");
            Assert.AreEqual(s1.Length, l());

        }
    }
}

