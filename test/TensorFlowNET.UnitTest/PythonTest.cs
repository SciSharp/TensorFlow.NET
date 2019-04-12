using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json.Linq;
using NumSharp;
using Tensorflow;
using Tensorflow.Util;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// Use as base class for test classes to get additional assertions
    /// </summary>
    public class PythonTest : Python
    {
        #region python compatibility layer
        protected PythonTest self { get => this; }
        protected object None {
            get { return null; }
        }
        #endregion

        #region pytest assertions

        public void assertItemsEqual(ICollection given, ICollection expected)
        {
            if (given is Hashtable && expected is Hashtable)
            {
                Assert.AreEqual(JObject.FromObject(expected).ToString(), JObject.FromObject(given).ToString());
                return;
            }
            Assert.IsNotNull(expected);
            Assert.IsNotNull(given);
            var e = expected.OfType<object>().ToArray();
            var g = given.OfType<object>().ToArray();
            Assert.AreEqual(e.Length, g.Length, $"The collections differ in length expected {e.Length} but got {g.Length}");
            for (int i = 0; i < e.Length; i++)
            {
                if (g[i] is NDArray && e[i] is NDArray)
                    assertItemsEqual((g[i] as NDArray).Array, (e[i] as NDArray).Array);
                else if (e[i] is ICollection && g[i] is ICollection)
                    assertEqual(g[i], e[i]);
                else 
                    Assert.AreEqual(e[i], g[i], $"Items differ at index {i}, expected {e[i]} but got {g[i]}");
            }
        }

        public void assertAllEqual(ICollection given, ICollection expected)
        {
            assertItemsEqual(given, expected);
        }


        public void assertEqual(object given, object expected)
        {
            if (given is NDArray && expected is NDArray)
            {
                assertItemsEqual((given as NDArray).Array, (expected as NDArray).Array);
                return;
            }
            if (given is Hashtable && expected is Hashtable)
            {
                Assert.AreEqual(JObject.FromObject(expected).ToString(), JObject.FromObject(given).ToString());
                return;
            }
            if (given is ICollection && expected is ICollection)
            {
                assertItemsEqual(given as ICollection, expected as ICollection);
                return;
            }
            Assert.AreEqual(expected, given);
        }

        public void assertEquals(object given, object expected)
        {
            assertEqual(given, expected);
        }

        public void assertIsNotNone(object given)
        {
            Assert.IsNotNull(given);
        }

        public void assertFalse(bool cond)
        {
            Assert.IsFalse(cond);
        }

        public void assertTrue(bool cond)
        {
            Assert.IsTrue(cond);
        }

        #endregion

        #region tensor evaluation

        protected object _eval_helper(Tensor[] tensors)
        {
            if (tensors == null)
                return null;
            //return nest.map_structure(self._eval_tensor, tensors);
            return null;
        }

        //def evaluate(self, tensors) :
        //  """Evaluates tensors and returns numpy values.

        //  Args:
        //    tensors: A Tensor or a nested list/tuple of Tensors.

        //  Returns:
        //    tensors numpy values.
        //  """
        //  if context.executing_eagerly():
        //    return self._eval_helper(tensors)
        //  else:
        //    sess = ops.get_default_session()
        //    if sess is None:
        //      with self.test_session() as sess:
        //        return sess.run(tensors)
        //    else:
        //      return sess.run(tensors)
        #endregion


    }
}
