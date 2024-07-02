using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Callbacks;
using Tensorflow.Keras.Engine.DataAdapters;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        /// <summary>
        /// Returns the loss value and metrics values for the model in test mode.
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
        public Dictionary<string, float> evaluate(NDArray x, NDArray y,
            int batch_size = -1,
            int verbose = 1,
            NDArray sample_weight = null,
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
                SampleWeight = sample_weight,
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

            return evaluate(data_handler, callbacks, is_val, test_function);
        }

        public Dictionary<string, float> evaluate(
            IEnumerable<Tensor> x, 
            Tensor y, 
            int verbose = 1,
            NDArray sample_weight = null,
            bool is_val = false)
        {
            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = new Tensors(x.ToArray()),
                Y = y,
                Model = this,
                SampleWeight = sample_weight,
                StepsPerExecution = _steps_per_execution
            });

            var callbacks = new CallbackList(new CallbackParams
            {
                Model = this,
                Verbose = verbose,
                Steps = data_handler.Inferredsteps
            });

            return evaluate(data_handler, callbacks, is_val, test_step_multi_inputs_function);
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

            Func<DataHandler, OwnedIterator, Dictionary<string, float>> testFunction;

            if (data_handler.DataAdapter.GetDataset().structure.Length > 2 ||
                data_handler.DataAdapter.GetDataset().FirstInputTensorCount > 1)
            {
                testFunction = test_step_multi_inputs_function;
            }
            else
            {
                testFunction = test_function;
            }

            return evaluate(data_handler, callbacks, is_val, testFunction);
        }

        /// <summary>
        /// Internal bare implementation of evaluate function.
        /// </summary>
        /// <param name="data_handler">Interations handling objects</param>
        /// <param name="callbacks"></param>
        /// <param name="test_func">The function to be called on each batch of data.</param>
        /// <param name="is_val">Whether it is validation or test.</param>
        /// <returns></returns>
        Dictionary<string, float> evaluate(DataHandler data_handler, CallbackList callbacks, bool is_val, Func<DataHandler, OwnedIterator, Dictionary<string, float>> test_func)
        {
            callbacks.on_test_begin();

            var logs = new Dictionary<string, float>();
            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                reset_metrics();
                foreach (var step in data_handler.steps())
                {
                    callbacks.on_test_batch_begin(step);
                    logs = test_func(data_handler, iterator);
                    var end_step = step + data_handler.StepIncrement;
                    if (!is_val)
                        callbacks.on_test_batch_end(end_step, logs);
                    GC.Collect();
                }
            }
            callbacks.on_test_end(logs);
            var results = new Dictionary<string, float>(logs);
            return results;
        }

        Dictionary<string, float> test_function(DataHandler data_handler, OwnedIterator iterator)
        {
            var data = iterator.next();
            var outputs = data.Length == 2 ? test_step(data_handler, data[0], data[1]) :
                            test_step(data_handler, data[0], data[1], data[2]);
            tf_with(ops.control_dependencies(new object[0]), ctl => _test_counter.assign_add(1));
            return outputs;
        }

        Dictionary<string, float> test_step_multi_inputs_function(DataHandler data_handler, OwnedIterator iterator)
        {
            var data = iterator.next();
            var x_size = data_handler.DataAdapter.GetDataset().FirstInputTensorCount;
            var outputs = data.Length == 2 ?
                            test_step(data_handler, new Tensors(data.Take(x_size).ToArray()), new Tensors(data.Skip(x_size).ToArray())) :
                            test_step(
                                data_handler,
                                new Tensors(data.Take(x_size).ToArray()),
                                new Tensors(data.Skip(x_size).Take(x_size).ToArray()),
                                new Tensors(data.Skip(2 * x_size).ToArray()));
            tf_with(ops.control_dependencies(new object[0]), ctl => _test_counter.assign_add(1));
            return outputs;
        }


        Dictionary<string, float> test_step(DataHandler data_handler, Tensors x, Tensors y)
        {
            (x,y) = data_handler.DataAdapter.Expand1d(x, y);

            var y_pred = Apply(x, training: false);

            var loss = compiled_loss.Call(y, y_pred);
            compiled_metrics.update_state(y, y_pred);
            return metrics.Select(x => (x.Name, x.result())).ToDictionary(x => x.Item1, x => (float)x.Item2);
        }

        Dictionary<string, float> test_step(DataHandler data_handler, Tensors x, Tensors y, Tensors sample_weight)
        {
            (x, y, sample_weight) = data_handler.DataAdapter.Expand1d(x, y, sample_weight);
            var y_pred = Apply(x, training: false);
            var loss = compiled_loss.Call(y, y_pred, sample_weight: sample_weight);
            compiled_metrics.update_state(y, y_pred);
            return metrics.Select(x => (x.Name, x.result())).ToDictionary(x => x.Item1, x => (float)x.Item2);
        }
    }
}
