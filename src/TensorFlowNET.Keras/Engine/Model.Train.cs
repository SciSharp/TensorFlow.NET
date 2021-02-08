using System.Collections.Generic;
using System.Linq;
using Tensorflow.Gradients;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        IEnumerable<(string, Tensor)> train_step_function(OwnedIterator iterator)
        {
            var data = iterator.next();
            var outputs = train_step(data[0], data[1]);
            tf_with(ops.control_dependencies(new object[0]), ctl => _train_counter.assign_add(1));
            return outputs;
        }

        /// <summary>
        /// The logic for one training step.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        List<(string, Tensor)> train_step(Tensor x, Tensor y)
        {
            (x, y) = data_handler.DataAdapter.Expand1d(x, y);
            using var tape = tf.GradientTape();
            var y_pred = Apply(x, training: true);
            var loss = compiled_loss.Call(y, y_pred);

            // For custom training steps, users can just write:
            // trainable_variables = self.trainable_variables
            // gradients = tape.gradient(loss, trainable_variables)
            // self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            // The _minimize call does a few extra steps unnecessary in most cases,
            // such as loss scaling and gradient clipping.
            _minimize(tape, optimizer, loss, trainable_variables);
            compiled_metrics.update_state(y, y_pred);

            return metrics.Select(x => (x.Name, x.result())).ToList();
        }

        void _minimize(GradientTape tape, OptimizerV2 optimizer, Tensor loss, List<IVariableV1> trainable_variables)
        {
            var gradients = tape.gradient(loss, trainable_variables);
            gradients = optimizer._aggregate_gradients(zip(gradients, trainable_variables));
            gradients = optimizer._clip_gradients(gradients);

            optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)),
                experimental_aggregate_gradients: false);
        }
    }
}
