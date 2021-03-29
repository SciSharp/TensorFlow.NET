using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    public abstract class DataAdapter
    {
        protected DataAdapterArgs args;
        protected IDatasetV2 dataset;

        public virtual bool CanHandle(Tensor x, Tensor y = null)
            => throw new NotImplementedException();

        public virtual IDatasetV2 GetDataset()
            => dataset;

        public virtual int GetSize()
            => throw new NotImplementedException("");

        public virtual (Tensor, Tensor) Expand1d(Tensor x, Tensor y)
        {
            if (x.TensorShape.ndim == 1)
                x = array_ops.expand_dims(x, axis: -1);
            if (y.TensorShape.ndim == 1)
                y = array_ops.expand_dims(y, axis: -1);
            return (x, y);
        }

        public virtual bool ShouldRecreateIterator()
        {
            return true;
        }
    }
}
