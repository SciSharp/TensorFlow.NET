using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Util;

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

        public virtual (Tensors, Tensors, Tensors) Expand1d(Tensors x, Tensors y, Tensors sample_weight)
        {
            for (int i = 0; i < x.Length; i++)
            {
                if (x[i].shape.ndim == 1)
                    x[i] = array_ops.expand_dims(x[i], axis: -1);
            }
            for (int i = 0; i < y.Length; i++)
            {
                if (y[i].shape.ndim == 1)
                    y[i] = array_ops.expand_dims(y[i], axis: -1);
            }
            for (int i = 0; i < sample_weight.Length; i++)
            {
                if (sample_weight[i].shape.ndim == 1)
                    sample_weight[i] = array_ops.expand_dims(sample_weight[i], axis: -1);
            }
            return (x, y, sample_weight);
        }

        public virtual bool ShouldRecreateIterator()
        {
            return true;
        }

        public static ((NDArray, NDArray, NDArray),ValidationDataPack) train_validation_split((NDArray, NDArray, NDArray) x_y_sample_weight, float validation_split)
        {
            var x = x_y_sample_weight.Item1;
            var y = x_y_sample_weight.Item2;
            var sample_weight = x_y_sample_weight.Item3;
            int train_count = Convert.ToInt32(x.dims[0] * (1 - validation_split));
            var train_x = x[new Slice(0, train_count)];
            var train_y = y[new Slice(0, train_count)];
            ValidationDataPack validation_data;
            if (sample_weight != null)
            {
                validation_data = (x[new Slice(train_count)], y[new Slice(train_count)], sample_weight[new Slice(train_count)]);
                sample_weight = sample_weight[new Slice(0, train_count)];
            }
            else
            {
                validation_data = (x[new Slice(train_count)], y[new Slice(train_count)]);
            }

            return ((train_x, train_y, sample_weight), validation_data);
        }

        public static ((IEnumerable<NDArray>, NDArray, NDArray), ValidationDataPack) train_validation_split((IEnumerable<NDArray>, NDArray, NDArray) x_y_sample_weight, float validation_split)
        {
            var x = x_y_sample_weight.Item1;
            var y = x_y_sample_weight.Item2;
            var sample_weight = x_y_sample_weight.Item3;
            int train_count = Convert.ToInt32(y.dims[0] * (1 - validation_split));
            var train_x = x.Select(x => x[new Slice(0, train_count)] as NDArray);
            var train_y = y[new Slice(0, train_count)];
            var val_x = x.Select(x => x[new Slice(train_count)] as NDArray);
            var val_y = y[new Slice(train_count)];

            ValidationDataPack validation_data;
            if (sample_weight != null)
            {
                validation_data = (val_x, val_y, sample_weight[new Slice(train_count)]);
                sample_weight = sample_weight[new Slice(0, train_count)];
            }
            else
            {
                validation_data = (val_x, val_y);
            }
            return ((train_x, train_y, sample_weight), validation_data);
        }
    }
}
