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
        /// <param name="callbacks"></param>
        /// <param name="verbose"></param>
        /// <param name="validation_split"></param>
        /// <param name="validation_data"></param>
        /// <param name="shuffle"></param>
        public ICallback fit(NDArray x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            float validation_split = 0f,
            (NDArray val_x, NDArray val_y)? validation_data = null,
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

            var train_x = x;
            var train_y = y;

            if (validation_split != 0f && validation_data == null)
            {
                int train_count = Convert.ToInt32(x.dims[0] * (1 - validation_split));
                train_x = x[new Slice(0, train_count)];
                train_y = y[new Slice(0, train_count)];
                validation_data = (val_x: x[new Slice(train_count)], val_y: y[new Slice(train_count)]);
            }

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

            return FitInternal(data_handler, epochs, verbose, callbackList: callbacks, validation_data: validation_data,
                    train_step_func: train_step_function);
        }

        public ICallback fit(IEnumerable<NDArray> x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            float validation_split = 0f,
            (IEnumerable<NDArray> val_x, NDArray val_y)? validation_data = null,
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

            var train_x = x;
            var train_y = y;
            if (validation_split != 0f && validation_data == null)
            {
                int train_count = Convert.ToInt32(y.dims[0] * (1 - validation_split));
                train_x = x.Select(x => x[new Slice(0, train_count)] as NDArray);
                train_y = y[new Slice(0, train_count)];
                var val_x = x.Select(x => x[new Slice(train_count)] as NDArray);
                var val_y = y[new Slice(train_count)];
                validation_data = (val_x, val_y);
            }


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
                return FitInternal(data_handler, epochs, verbose, callbackList: callbacks, validation_data: validation_data,
                    train_step_func: train_step_multi_inputs_function);
            }
            else
            {
                return FitInternal(data_handler, epochs, verbose, callbackList: callbacks, validation_data: validation_data,
                    train_step_func: train_step_function);
            }
        }

        public History fit(IDatasetV2 dataset, 
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            IDatasetV2 validation_data = null,
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


            return FitInternal(data_handler, epochs, verbose, callbacks, validation_data: validation_data,
                    train_step_func: train_step_function);
        }

        History FitInternal(DataHandler data_handler, int epochs, int verbose, List<ICallback> callbackList, IDatasetV2 validation_data, 
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
          
            if (callbackList != null)
            {
                foreach(var callback in callbackList)
                    callbacks.callbacks.add(callback);
            }
            
            callbacks.on_train_begin();

            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                reset_metrics();
                callbacks.on_epoch_begin(epoch);
                // data_handler.catch_stop_iteration();
                var logs = new Dictionary<string, float>();
                long End_step = 0;
                foreach (var step in data_handler.steps())
                {
                    callbacks.on_train_batch_begin(step);
                    logs = train_step_func(data_handler, iterator);
                    var end_step = step + data_handler.StepIncrement;
                    End_step = end_step;
                    callbacks.on_train_batch_end(end_step, logs);
                }

                if (validation_data != null)
                {
                    var val_logs = evaluate(validation_data);
                    foreach(var log in val_logs)
                    {
                        logs["val_" + log.Key] = log.Value;
                    }
                    callbacks.on_train_batch_end(End_step, logs);
                }


                callbacks.on_epoch_end(epoch, logs);

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            return callbacks.History;
        }

        History FitInternal(DataHandler data_handler, int epochs, int verbose, List<ICallback> callbackList, (NDArray, NDArray)? validation_data,
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

            if (callbackList != null)
            {
                foreach (var callback in callbackList)
                    callbacks.callbacks.add(callback);
            }

            callbacks.on_train_begin();

            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                reset_metrics();
                callbacks.on_epoch_begin(epoch);
                // data_handler.catch_stop_iteration();
                var logs = new Dictionary<string, float>();
                long End_step = 0;
                foreach (var step in data_handler.steps())
                {
                    callbacks.on_train_batch_begin(step);
                    logs = train_step_func(data_handler, iterator);
                    var end_step = step + data_handler.StepIncrement;
                    End_step = end_step;
                    callbacks.on_train_batch_end(end_step, logs);
                }

                if (validation_data != null)
                {
                    // Because evaluate calls call_test_batch_end, this interferes with our output on the screen
                    // so we need to pass a is_val parameter to stop on_test_batch_end
                    var val_logs = evaluate(validation_data.Value.Item1, validation_data.Value.Item2, is_val:true);
                    foreach (var log in val_logs)
                    {
                        logs["val_" + log.Key] = log.Value;
                    }
                    // because after evaluate, logs add some new log which we need to print
                    callbacks.on_train_batch_end(End_step, logs);
                }

                callbacks.on_epoch_end(epoch, logs);

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            return callbacks.History;
        }

        History FitInternal(DataHandler data_handler, int epochs, int verbose, List<ICallback> callbackList, (IEnumerable<Tensor>, NDArray)? validation_data,
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

            if (callbackList != null)
            {
                foreach (var callback in callbackList)
                    callbacks.callbacks.add(callback);
            }

            callbacks.on_train_begin();

            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                reset_metrics();
                callbacks.on_epoch_begin(epoch);
                // data_handler.catch_stop_iteration();
                var logs = new Dictionary<string, float>();
                long End_step = 0;
                foreach (var step in data_handler.steps())
                {
                    callbacks.on_train_batch_begin(step);
                    logs = train_step_func(data_handler, iterator);
                    var end_step = step + data_handler.StepIncrement;
                    End_step = end_step;
                    callbacks.on_train_batch_end(end_step, logs);
                }

                if (validation_data != null)
                {
                    var val_logs = evaluate(validation_data.Value.Item1, validation_data.Value.Item2);
                    foreach (var log in val_logs)
                    {
                        logs["val_" + log.Key] = log.Value;
                        callbacks.on_train_batch_end(End_step, logs);
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
