using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using System.Diagnostics;
using Tensorflow.Keras.Callbacks;
using Tensorflow.Util;
using OneOf;

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
        /// <param name="callbacks"></param>
        /// <param name="validation_split"></param>
        /// <param name="validation_data"></param>
        /// <param name="shuffle"></param>
        /// <param name="class_weight"></param>
        /// <param name="sample_weight"></param>
        /// <param name="initial_epoch"></param>
        /// <param name="max_queue_size"></param>
        /// <param name="workers"></param>
        /// <param name="use_multiprocessing"></param>
        /// <returns></returns>
        /// <exception cref="InvalidArgumentError"></exception>
        public ICallback fit(NDArray x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            float validation_split = 0f,
            ValidationDataPack validation_data = null,
            int validation_step = 10,
            bool shuffle = true,
            Dictionary<int, float> class_weight = null,
            NDArray sample_weight = null,
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

            // The default dtype in NDArray is double, so we need to cast sample_weight to float to mul with loss which's dtype is float.
            sample_weight = sample_weight?.astype(TF_DataType.TF_FLOAT);

            if (validation_split != 0f && validation_data == null)
            {
                ((x, y, sample_weight), validation_data) = DataAdapter.train_validation_split((x, y, sample_weight), validation_split);
            }

            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = x,
                Y = y,
                SampleWeight = sample_weight,
                BatchSize = batch_size,
                InitialEpoch = initial_epoch,
                Epochs = epochs,
                Shuffle = shuffle,
                ClassWeight = class_weight,
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
            ValidationDataPack validation_data = null,
            bool shuffle = true,
            Dictionary<int, float> class_weight = null,
            NDArray sample_weight = null,
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

            sample_weight = sample_weight?.astype(TF_DataType.TF_FLOAT);

            if (validation_split != 0f && validation_data == null)
            {
                ((x, y, sample_weight), validation_data) = DataAdapter.train_validation_split((x, y, sample_weight), validation_split);
            }


            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = new Tensors(x.ToArray()),
                Y = y,
                SampleWeight = sample_weight,
                BatchSize = batch_size,
                InitialEpoch = initial_epoch,
                Epochs = epochs,
                Shuffle = shuffle,
                ClassWeight = class_weight,
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

        public ICallback fit(IDatasetV2 dataset, 
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            IDatasetV2 validation_data = null,
            int validation_step = 10,
            bool shuffle = true,
            Dictionary<int, float> class_weight = null,
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
                ClassWeight = class_weight,
                MaxQueueSize = max_queue_size,
                Workers = workers,
                UseMultiprocessing = use_multiprocessing,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            Func<DataHandler, OwnedIterator, Dictionary<string, float>> trainStepFunction;

            if (data_handler.DataAdapter.GetDataset().structure.Length > 2 ||
                data_handler.DataAdapter.GetDataset().FirstInputTensorCount > 1)
            {
                trainStepFunction = train_step_multi_inputs_function;
            }
            else
            {
                trainStepFunction = train_step_function;
            }

            return FitInternal(data_handler, epochs, validation_step, verbose, callbacks, validation_data: validation_data,
                    train_step_func: trainStepFunction);
        }

        History FitInternal(DataHandler data_handler, int epochs, int validation_step, int verbose, List<ICallback> callbackList, IDatasetV2 validation_data, 
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
                    GC.Collect();
                }

                if (validation_data != null)
                {
                    if (validation_step > 0 && epoch ==0 || (epoch) % validation_step != 0)
                        continue;
                    
                    var val_logs = evaluate(validation_data);
                    foreach(var log in val_logs)
                    {
                        logs["val_" + log.Key] = log.Value;
                    }
                    callbacks.on_train_batch_end(End_step, logs);
                }

                GC.Collect();

                callbacks.on_epoch_end(epoch, logs);

                if (stop_training)
                {
                    break;
                }
            }

            return callbacks.History;
        }

        History FitInternal(DataHandler data_handler, int epochs, int verbose, List<ICallback> callbackList, ValidationDataPack validation_data,
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
                    GC.Collect();
                }

                if (validation_data != null)
                {
                    NDArray val_x;
                    NDArray[] val_x_array;
                    NDArray val_y;
                    NDArray val_sample_weight;
                    Dictionary<string, float> val_logs;
                    if (!validation_data.val_x_is_array)
                    {
                        (val_x, val_y, val_sample_weight) = validation_data;
                        // Because evaluate calls call_test_batch_end, this interferes with our output on the screen
                        // so we need to pass a is_val parameter to stop on_test_batch_end
                        val_logs = evaluate(val_x, val_y, sample_weight: val_sample_weight, is_val: true);

                    }
                    else
                    {
                        (val_x_array, val_y, val_sample_weight, _) = validation_data;
                         val_logs = evaluate(val_x_array, val_y, sample_weight: val_sample_weight, is_val: true);
                    }
                    foreach (var log in val_logs)
                    {
                        logs["val_" + log.Key] = log.Value;
                    }
                    // because after evaluate, logs add some new log which we need to print
                    callbacks.on_train_batch_end(End_step, logs);
                }

                callbacks.on_epoch_end(epoch, logs);

                GC.Collect();
                if (stop_training)
                {
                    break;
                }
            }

            return callbacks.History;
        }

    }
}
