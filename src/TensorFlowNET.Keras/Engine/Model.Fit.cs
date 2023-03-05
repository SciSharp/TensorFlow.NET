using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using System.Diagnostics;
using Tensorflow.Keras.Callbacks;
using System.Data;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        /// <summary>
        /// Trains the model for a fixed number of epochs (iterations on a dataset).
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="batch_size"></param>
        /// <param name="epochs"></param>
        /// <param name="verbose"></param>
        /// <param name="validation_split"></param>
        /// <param name="shuffle"></param>
        public ICallback fit(NDArray x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            float validation_split = 0f,
            bool shuffle = true,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
        {
            if (x.dims[0] != y.dims[0])
            {
                throw new InvalidArgumentError(
                    $"The array x and y should have same value at dim 0, but got {x.dims[0]} and {y.dims[0]}");
            }
            int train_count = Convert.ToInt32(x.dims[0] * (1 - validation_split));
            var train_x = x[new Slice(0, train_count)];
            var train_y = y[new Slice(0, train_count)];
            var val_x = x[new Slice(train_count)];
            var val_y = y[new Slice(train_count)];

            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = train_x,
                Y = train_y,
                BatchSize = batch_size,
                InitialEpoch = initial_epoch,
                Epochs = epochs,
                Shuffle = shuffle,
                MaxQueueSize = max_queue_size,
                Workers = workers,
                UseMultiprocessing = use_multiprocessing,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            return FitInternal(data_handler, epochs, verbose, validation_data: null,
                    train_step_func: train_step_function);
        }

        public ICallback fit(IEnumerable<NDArray> x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            float validation_split = 0f,
            bool shuffle = true,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
        {
            foreach(var tx in x)
            {
                if (tx.dims[0] != y.dims[0])
                {
                    throw new InvalidArgumentError(
                        $"The array x and y should have same value at dim 0, but got {tx.dims[0]} and {y.dims[0]}");
                }
            }
            int train_count = Convert.ToInt32(y.dims[0] * (1 - validation_split));
            
            var train_x = x.Select(x => x[new Slice(0, train_count)] as Tensor);
            var train_y = y[new Slice(0, train_count)];
            var val_x = x.Select(x => x[new Slice(train_count)] as Tensor);
            var val_y = y[new Slice(train_count)];

            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = new Tensors(train_x),
                Y = train_y,
                BatchSize = batch_size,
                InitialEpoch = initial_epoch,
                Epochs = epochs,
                Shuffle = shuffle,
                MaxQueueSize = max_queue_size,
                Workers = workers,
                UseMultiprocessing = use_multiprocessing,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            if (data_handler.DataAdapter.GetDataset().structure.Length > 2 ||
                data_handler.DataAdapter.GetDataset().FirstInputTensorCount > 1)
            {
                return FitInternal(data_handler, epochs, verbose, validation_data: null,
                    train_step_func: train_step_multi_inputs_function);
            }
            else
            {
                return FitInternal(data_handler, epochs, verbose, validation_data: null,
                    train_step_func: train_step_function);
            }
        }

        public History fit(IDatasetV2 dataset, 
            IDatasetV2 validation_data = null,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            float validation_split = 0f,
            bool shuffle = true,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
        {
            var data_handler = new DataHandler(new DataHandlerArgs
            {
                Dataset = dataset,
                BatchSize = batch_size,
                InitialEpoch = initial_epoch,
                Epochs = epochs,
                Shuffle = shuffle,
                MaxQueueSize = max_queue_size,
                Workers = workers,
                UseMultiprocessing = use_multiprocessing,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            return FitInternal(data_handler, epochs, verbose, validation_data: validation_data,
                    train_step_func: train_step_function);
        }

        History FitInternal(DataHandler data_handler, int epochs, int verbose, IDatasetV2 validation_data, 
            Func<DataHandler, OwnedIterator, Dictionary<string, float>> train_step_func)
        {
            stop_training = false;
            _train_counter.assign(0);
            var callbacks = new CallbackList(new CallbackParams
            {
                Model = this,
                Verbose = verbose,
                Epochs = epochs,
                Steps = data_handler.Inferredsteps
            });
            callbacks.on_train_begin();

            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                reset_metrics();
                callbacks.on_epoch_begin(epoch);
                // data_handler.catch_stop_iteration();
                var logs = new Dictionary<string, float>();
                foreach (var step in data_handler.steps())
                {
                    callbacks.on_train_batch_begin(step);
                    logs = train_step_func(data_handler, iterator);
                    var end_step = step + data_handler.StepIncrement;
                    callbacks.on_train_batch_end(end_step, logs);
                }

                if (validation_data != null)
                {
                    var val_logs = evaluate(validation_data);
                    foreach(var log in val_logs)
                    {
                        logs["val_" + log.Key] = log.Value;
                    }
                }

                callbacks.on_epoch_end(epoch, logs);

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            return callbacks.History;
        }
    }
}
