using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        /// <summary>
        /// Returns the loss value & metrics values for the model in test mode.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="batch_size"></param>
        /// <param name="verbose"></param>
        /// <param name="steps"></param>
        /// <param name="max_queue_size"></param>
        /// <param name="workers"></param>
        /// <param name="use_multiprocessing"></param>
        /// <param name="return_dict"></param>
        public void evaluate(NDArray x, NDArray y,
            int batch_size = -1,
            int verbose = 1,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false,
            bool return_dict = false)
        {
            data_handler = new DataHandler(new DataHandlerArgs
            {
                X = x,
                Y = y,
                BatchSize = batch_size,
                StepsPerEpoch = steps,
                InitialEpoch = 0,
                Epochs = 1,
                MaxQueueSize = max_queue_size,
                Workers = workers,
                UseMultiprocessing = use_multiprocessing,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            Console.WriteLine($"Testing...");
            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                // reset_metrics();
                // callbacks.on_epoch_begin(epoch)
                // data_handler.catch_stop_iteration();
                IEnumerable<(string, Tensor)> results = null;
                foreach (var step in data_handler.steps())
                {
                    // callbacks.on_train_batch_begin(step)
                    results = test_function(iterator);
                }
                Console.WriteLine($"iterator: {epoch + 1}, " + string.Join(", ", results.Select(x => $"{x.Item1}: {(float)x.Item2}")));
            }
        }

        IEnumerable<(string, Tensor)> test_function(OwnedIterator iterator)
        {
            var data = iterator.next();
            var outputs = test_step(data[0], data[1]);
            tf_with(ops.control_dependencies(new object[0]), ctl => _test_counter.assign_add(1));
            return outputs;
        }

        List<(string, Tensor)> test_step(Tensor x, Tensor y)
        {
            (x, y) = data_handler.DataAdapter.Expand1d(x, y);
            var y_pred = Apply(x, training: false);
            var loss = compiled_loss.Call(y, y_pred);

            compiled_metrics.update_state(y, y_pred);

            return metrics.Select(x => (x.Name, x.result())).ToList();
        }
    }
}
