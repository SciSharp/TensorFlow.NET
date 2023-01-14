using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using System.Diagnostics;

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

            FitInternal(epochs, verbose);
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

            FitInternal(epochs, verbose);
        }

        void FitInternal(int epochs, int verbose)
        {
            stop_training = false;
            _train_counter.assign(0);
            Stopwatch sw = new Stopwatch();
            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                reset_metrics();
                on_epoch_begin(epoch, epochs);
                // data_handler.catch_stop_iteration();
                foreach (var step in data_handler.steps())
                {
                    sw.Start();
                    var results = train_step_function(iterator);
                    sw.Stop();
                    on_train_batch_begin(verbose, step, sw.ElapsedMilliseconds, results);

                    // recycle memory more frequency
                    if (sw.ElapsedMilliseconds > 100)
                    {
                        GC.Collect();
                    }
                    sw.Reset();
                }
                Console.WriteLine();

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        void on_epoch_begin(int epoch, int epochs)
        {
            Binding.tf_output_redirect.WriteLine($"Epoch: {epoch + 1:D3}/{epochs:D3}");
        }

        void on_train_batch_begin(int verbose, long step, long elapse, IEnumerable<(string, Tensor)> results)
        {
            if (verbose == 1)
            {
                var result_pairs = string.Join(", ", results.Select(x => $"{x.Item1}: {(float)x.Item2:F6}"));

                var progress = "";
                for (int i = 0; i < step + 1; i++)
                    for (int j = 0; j < 30 / data_handler.Inferredsteps; j++)
                        progress += "=";
                progress += ">";

                var remaining = "";
                for (int i = 1; i < 30 - progress.Length; i++)
                    remaining += ".";

                Binding.tf_output_redirect.Write($"{step + 1:D4}/{data_handler.Inferredsteps:D4} [{progress}{remaining}] - {elapse}ms/step {result_pairs}");
                if (!Console.IsOutputRedirected)
                {
                    Console.CursorLeft = 0;
                }
            }
        }
    }
}
