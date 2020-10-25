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
        IDataAdapter _adapter;

        public DataHandler(DataHandlerArgs args)
        {
            this.args = args;

            _adapter = new TensorLikeDataAdapter(new TensorLikeDataAdapterArgs
            {
                X = args.X,
                Y = args.Y,
                BatchSize = args.BatchSize,
                Steps = args.StepsPerEpoch,
                Epochs = args.Epochs - args.InitialEpoch,
                Shuffle = args.Shuffle,
                MaxQueueSize = args.MaxQueueSize,
                Worker = args.Workers,
                UseMultiprocessing = args.UseMultiprocessing,
                Model = args.Model
            });
        }

        Tensor _infer_steps(IDatasetV2 dataset)
        {
            throw new NotImplementedException("");
        }
    }
}
