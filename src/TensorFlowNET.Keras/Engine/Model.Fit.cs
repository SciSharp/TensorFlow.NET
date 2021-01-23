using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;

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
        public void fit(NDArray x, NDArray y,
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
            int train_count = Convert.ToInt32(x.shape[0] * (1 - validation_split));
            var train_x = x[new Slice(0, train_count)];
            var train_y = y[new Slice(0, train_count)];
            var val_x = x[new Slice(train_count)];
            var val_y = y[new Slice(train_count)];

            data_handler = new DataHandler(new DataHandlerArgs
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

            FitInternal(epochs);
        }

        public void fit(IDatasetV2 dataset, 
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
            data_handler = new DataHandler(new DataHandlerArgs
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

            FitInternal(epochs);
        }

        void FitInternal(int epochs)
        {
            stop_training = false;
            _train_counter.assign(0);
            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                // reset_metrics();
                // callbacks.on_epoch_begin(epoch)
                // data_handler.catch_stop_iteration();
                foreach (var step in data_handler.steps())
                {
                    // callbacks.on_train_batch_begin(step)
                    var results = train_step_function(iterator);
                    var result_pairs = string.Join(", ", results.Select(x => $"{x.Item1}: {(float)x.Item2:F6}"));
                    Console.WriteLine($"Epoch: {epoch + 1:D3}/{epochs:D3}, Step: {step + 1:D4}/{data_handler.Inferredsteps:D4}, {result_pairs}");
                }

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }
    }
}
