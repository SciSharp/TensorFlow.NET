using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using static Tensorflow.Binding;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow;
using Tensorflow.Keras.Callbacks;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        protected Dictionary<string, float> evaluate(CallbackList callbacks, DataHandler data_handler, bool is_val)
        {
            callbacks.on_test_begin();

            //Dictionary<string, float>? logs = null;
            var logs = new Dictionary<string, float>();
            int x_size = data_handler.DataAdapter.GetDataset().FirstInputTensorCount;
            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                reset_metrics();
                callbacks.on_epoch_begin(epoch);
                // data_handler.catch_stop_iteration();

                foreach (var step in data_handler.steps())
                {
                    callbacks.on_test_batch_begin(step);

                    var data = iterator.next();

                    logs = train_step(data_handler, new Tensors(data.Take(x_size)), new Tensors(data.Skip(x_size)));
                    tf_with(ops.control_dependencies(Array.Empty<object>()), ctl => _test_counter.assign_add(1));

                    var end_step = step + data_handler.StepIncrement;

                    if (!is_val)
                        callbacks.on_test_batch_end(end_step, logs);
                }
            }

            return logs;
        }

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
        /// <param name="is_val"></param>
        public Dictionary<string, float> evaluate(Tensor x, Tensor y,
            int batch_size = -1,
            int verbose = 1,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false,
            bool return_dict = false,
            bool is_val = false
            )
        {
            if (x.dims[0] != y.dims[0])
            {
                throw new InvalidArgumentError(
                    $"The array x and y should have same value at dim 0, but got {x.dims[0]} and {y.dims[0]}");
            }
            var data_handler = new DataHandler(new DataHandlerArgs
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

            var callbacks = new CallbackList(new CallbackParams
            {
                Model = this,
                Verbose = verbose,
                Steps = data_handler.Inferredsteps
            });

            return evaluate(callbacks, data_handler, is_val);
        }

        public Dictionary<string, float> evaluate(IEnumerable<Tensor> x, Tensor y, int verbose = 1, bool is_val = false)
        {
            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = new Tensors(x),
                Y = y,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            var callbacks = new CallbackList(new CallbackParams
            {
                Model = this,
                Verbose = verbose,
                Steps = data_handler.Inferredsteps
            });

            return evaluate(callbacks, data_handler, is_val);
        }


        public Dictionary<string, float> evaluate(IDatasetV2 x, int verbose = 1, bool is_val = false)
        {
            var data_handler = new DataHandler(new DataHandlerArgs
            {
                Dataset = x,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            var callbacks = new CallbackList(new CallbackParams
            {
                Model = this,
                Verbose = verbose,
                Steps = data_handler.Inferredsteps
            });

            return evaluate(callbacks, data_handler, is_val);
        }
    }
}