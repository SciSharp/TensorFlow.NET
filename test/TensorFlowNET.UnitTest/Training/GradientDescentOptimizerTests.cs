using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Security.AccessControl;
using Tensorflow.NumPy;
using TensorFlowNET.UnitTest;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest.Optimizers
{
    [TestClass]
    public class GradientDescentOptimizerTest : PythonTest
    {
        private void TestBasicGeneric<T>() where T : struct
        {
            var dtype = Type.GetTypeCode(typeof(T)) switch
            {
                TypeCode.Single => np.float32,
                TypeCode.Double => np.float64,
                _ => throw new NotImplementedException(),
            };

            // train.GradientDescentOptimizer is V1 only API.
            tf.Graph().as_default();
            using (self.cached_session())
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

                var global_variables = variables.global_variables_initializer();
                self.evaluate<T>(global_variables);
                // Fetch params to validate initial values
                // TODO: use self.evaluate<T[]> instead of self.evaluate<double[]>
                self.assertAllCloseAccordingToType(new double[] { 1.0, 2.0 }, self.evaluate<double[]>(var0));
                self.assertAllCloseAccordingToType(new double[] { 3.0, 4.0 }, self.evaluate<double[]>(var1));
                // Run 1 step of sgd
                sgd_op.run();
                // Validate updated params
                self.assertAllCloseAccordingToType(
                    new double[] { 1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1 },
                    self.evaluate<double[]>(var0));
                self.assertAllCloseAccordingToType(
                    new double[] { 3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01 },
                    self.evaluate<double[]>(var1));
                // TODO: self.assertEqual(0, len(optimizer.variables()));
            }
        }

        [TestMethod]
        public void TestBasic()
        {
            //TODO: add np.half
            TestBasicGeneric<float>();
            TestBasicGeneric<double>();
        }


    }
}
