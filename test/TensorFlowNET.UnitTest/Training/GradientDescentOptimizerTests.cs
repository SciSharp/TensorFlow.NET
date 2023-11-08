using Microsoft.VisualStudio.TestPlatform.Utilities;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using Tensorflow.NumPy;
using TensorFlowNET.UnitTest;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest.Optimizers
{
    [TestClass]
    public class GradientDescentOptimizerTest : PythonTest
    {
        private static TF_DataType GetTypeForNumericType<T>() where T : struct
        {
            return Type.GetTypeCode(typeof(T)) switch
            {
                TypeCode.Single => np.float32,
                TypeCode.Double => np.float64,
                _ => throw new NotImplementedException(),
            };
        }

        private void TestBasic<T>() where T : struct
        {
            var dtype = GetTypeForNumericType<T>();

            // train.GradientDescentOptimizer is V1 only API.
            tf.Graph().as_default();
            using (var sess = self.cached_session())
            {
                var var0 = tf.Variable(new[] { 1.0, 2.0 }, dtype: dtype);
                var var1 = tf.Variable(new[] { 3.0, 4.0 }, dtype: dtype);
                var grads0 = tf.constant(new[] { 0.1, 0.1 }, dtype: dtype);
                var grads1 = tf.constant(new[] { 0.01, 0.01 }, dtype: dtype);
                var optimizer = tf.train.GradientDescentOptimizer(3.0f);
                var grads_and_vars = new[] {
                    Tuple.Create(grads0, var0 as IVariableV1),
                    Tuple.Create(grads1, var1 as IVariableV1)
                };
                var sgd_op = optimizer.apply_gradients(grads_and_vars);

                var global_variables = tf.global_variables_initializer();
                sess.run(global_variables);

                var initialVar0 = sess.run(var0);
                var initialVar1 = sess.run(var1);
                // Fetch params to validate initial values
                self.assertAllCloseAccordingToType(new[] { 1.0, 2.0 }, self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(new[] { 3.0, 4.0 }, self.evaluate<T[]>(var1));
                // Run 1 step of sgd
                sgd_op.run();
                // Validate updated params
                self.assertAllCloseAccordingToType(
                    new[] { 1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1 },
                    self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(
                    new[] { 3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01 },
                    self.evaluate<T[]>(var1));
                // TODO: self.assertEqual(0, len(optimizer.variables()));
            }
        }

        [TestMethod]
        public void TestBasic()
        {
            //TODO: add np.half
            TestBasic<float>();
            TestBasic<double>();
        }

    }
}
