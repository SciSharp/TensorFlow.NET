using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    /// <summary>
    /// Handles iterating over epoch-level `tf.data.Iterator` objects.
    /// </summary>
    public class DataHandler
    {
        DataHandlerArgs args;

        Tensor x => args.X;
        Tensor y => args.Y;
        int batch_size => args.BatchSize;
        int steps_per_epoch => args.StepsPerEpoch;
        int initial_epoch => args.InitialEpoch;
        int epochs => args.Epochs;
        bool shuffle => args.Shuffle;
        int max_queue_size => args.MaxQueueSize;
        int workers => args.Workers;
        bool use_multiprocessing => args.UseMultiprocessing;
        Model model => args.Model;
        IVariableV1 steps_per_execution => args.StepsPerExecution;

        public DataHandler(DataHandlerArgs args)
        {
            this.args = args;

            var adapter_cls = new TensorLikeDataAdapter(new TensorLikeDataAdapterArgs { });
        }
    }
}
