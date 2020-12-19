using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    public class DatasetAdapter : IDataAdapter
    {
        DataAdapterArgs args;
        IDatasetV2 _dataset => args.Dataset;
        public DatasetAdapter(DataAdapterArgs args)
        {
            this.args = args;
        }

        public bool CanHandle(Tensor x, Tensor y = null)
        {
            throw new NotImplementedException();
        }

        public IDatasetV2 GetDataset()
            => _dataset;

        public int GetSize()
            => -1;

        public (Tensor, Tensor) Expand1d(Tensor x, Tensor y)
        {
            if (y.TensorShape.ndim == 1)
                y = array_ops.expand_dims(y, axis: -1);
            return (x, y);
        }
    }
}
