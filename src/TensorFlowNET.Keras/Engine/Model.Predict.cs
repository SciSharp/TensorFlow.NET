using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using static Tensorflow.Binding;
using Tensorflow.Keras.Callbacks;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        public Tensors predict(IDatasetV2 dataset,
            int batch_size = -1,
            int verbose = 0,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
        {
            var data_handler = new DataHandler(new DataHandlerArgs
            {
                Dataset = dataset,
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

            return PredictInternal(data_handler, verbose);
        }

        /// <summary>
        /// Generates output predictions for the input samples.
        /// </summary>
        /// <param name="x">Input samples</param>
        /// <param name="batch_size">Number of samples per batch</param>
        /// <param name="verbose">Verbosity mode</param>
        /// <param name="steps">
        /// Total number of steps (batches of samples)
        /// before declaring the prediction round finished.
        /// </param>
        /// <param name="max_queue_size"></param>
        /// <param name="workers"></param>
        /// <param name="use_multiprocessing"></param>
        /// <returns></returns>
        public Tensors predict(Tensor x,
            int batch_size = -1,
            int verbose = 0,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
        {
            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = x,
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

            return PredictInternal(data_handler, verbose);
        }

        Tensors PredictInternal(DataHandler data_handler, int verbose)
        {
            var callbacks = new CallbackList(new CallbackParams
            {
                Model = this,
                Verbose = verbose,
                Epochs = 1,
                Steps = data_handler.Inferredsteps
            });

            Tensor batch_outputs = null;
            _predict_counter.assign(0);
            callbacks.on_predict_begin();
            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                foreach (var step in data_handler.steps())
                {
                    callbacks.on_predict_batch_begin(step);
                    var tmp_batch_outputs = run_predict_step(iterator);
                    if (batch_outputs == null)
                    {
                        batch_outputs = tmp_batch_outputs[0];
                    }
                    else
                    {
                        batch_outputs = tf.concat(new Tensor[] { batch_outputs, tmp_batch_outputs[0] }, axis: 0);
                    }

                    var end_step = step + data_handler.StepIncrement;
                    callbacks.on_predict_batch_end(end_step, new Dictionary<string, Tensors> { { "outputs", batch_outputs } });
                }
            }

            callbacks.on_predict_end();

            return batch_outputs;
        }

        Tensors run_predict_step(OwnedIterator iterator)
        {
            var data = iterator.next();
            var outputs = predict_step(data[0]);
            tf_with(ops.control_dependencies(new object[0]), ctl => _predict_counter.assign_add(1));
            return outputs;
        }

        Tensors predict_step(Tensor data)
        {
            return Apply(data, training: false);
        }
    }
}
