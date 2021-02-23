using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class RandomTest
    {
        /// <summary>
        /// Test the function of setting random seed
        /// This will help regenerate the same result
        /// </summary>
        [TestMethod]
        public void TFRandomSeedTest()
        {
            var initValue = np.arange(6).reshape(3, 2);
            tf.set_random_seed(1234);
            var a1 = tf.random_uniform(1);
            var b1 = tf.random_shuffle(tf.constant(initValue));

            // This part we consider to be a refresh
            tf.set_random_seed(10);
            tf.random_uniform(1);
            tf.random_shuffle(tf.constant(initValue));

            tf.set_random_seed(1234);
            var a2 = tf.random_uniform(1);
            var b2 = tf.random_shuffle(tf.constant(initValue));
            Assert.IsTrue(a1.numpy().array_equal(a2.numpy()));
            Assert.IsTrue(b1.numpy().array_equal(b2.numpy()));
        }
        
        /// <summary>
        /// compare to Test above, seed is also added in params
        /// </summary>
        [TestMethod, Ignore]
        public void TFRandomSeedTest2()
        {
            var initValue = np.arange(6).reshape(3, 2);
            tf.set_random_seed(1234);
            var a1 = tf.random_uniform(1, seed:1234);
            var b1 = tf.random_shuffle(tf.constant(initValue), seed: 1234);

            // This part we consider to be a refresh
            tf.set_random_seed(10);
            tf.random_uniform(1);
            tf.random_shuffle(tf.constant(initValue));

            tf.set_random_seed(1234);
            var a2 = tf.random_uniform(1);
            var b2 = tf.random_shuffle(tf.constant(initValue));
            Assert.IsTrue(a1.numpy().array_equal(a2.numpy()));
            Assert.IsTrue(b1.numpy().array_equal(b2.numpy()));
        }

        /// <summary>
        /// This part we use funcs in tf.random rather than only tf
        /// </summary>
        [TestMethod]
        public void TFRandomRaodomSeedTest()
        {
            tf.set_random_seed(1234);
            var a1 = tf.random.normal(1);
            var b1 = tf.random.truncated_normal(1);

            // This part we consider to be a refresh
            tf.set_random_seed(10);
            tf.random.normal(1);
            tf.random.truncated_normal(1);

            tf.set_random_seed(1234);
            var a2 = tf.random.normal(1);
            var b2 = tf.random.truncated_normal(1);

            Assert.IsTrue(a1.numpy().array_equal(a2.numpy()));
            Assert.IsTrue(b1.numpy().array_equal(b2.numpy()));
        }

        /// <summary>
        /// compare to Test above, seed is also added in params
        /// </summary>
        [TestMethod, Ignore]
        public void TFRandomRaodomSeedTest2()
        {
            tf.set_random_seed(1234);
            var a1 = tf.random.normal(1, seed:1234);
            var b1 = tf.random.truncated_normal(1);

            // This part we consider to be a refresh
            tf.set_random_seed(10);
            tf.random.normal(1);
            tf.random.truncated_normal(1);

            tf.set_random_seed(1234);
            var a2 = tf.random.normal(1, seed:1234);
            var b2 = tf.random.truncated_normal(1, seed:1234);

            Assert.IsTrue(a1.numpy().array_equal(a2.numpy()));
            Assert.IsTrue(b1.numpy().array_equal(b2.numpy()));
        }
    }
}