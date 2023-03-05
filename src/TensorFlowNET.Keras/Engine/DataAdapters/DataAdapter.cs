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

        public virtual bool CanHandle(Tensors x, Tensors y = null)
            => throw new NotImplementedException();

        public virtual IDatasetV2 GetDataset()
            => dataset;

        public virtual int GetSize()
            => throw new NotImplementedException("");

        public virtual (Tensors, Tensors) Expand1d(Tensors x, Tensors y)
        {
            for(int i = 0; i < x.Length; i++)
            {
                if (x[i].shape.ndim == 1)
                    x[i] = array_ops.expand_dims(x[i], axis: -1);
            }
            for (int i = 0; i < y.Length; i++)
            {
                if (y[i].shape.ndim == 1)
                    y[i] = array_ops.expand_dims(y[i], axis: -1);
            }
            return (x, y);
        }

        public virtual bool ShouldRecreateIterator()
        {
            return true;
        }
    }
}
