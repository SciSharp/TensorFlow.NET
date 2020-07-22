using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework.Models;

namespace Tensorflow
{
    /// <summary>
    /// Abstract class representing a dataset with no inputs.
    /// </summary>
    public class DatasetV2 : IDatasetV2
    {
        protected dataset_ops ops = new dataset_ops();
        public Tensor variant_tensor { get; set; }

        public TensorSpec[] _structure { get; set; }

        public TensorShape[] output_shapes => _structure.Select(x => x.shape).ToArray();
        
        public TF_DataType[] output_types => _structure.Select(x => x.dtype).ToArray();
        
        public TensorSpec[] element_spec => _structure;

        public IDatasetV2 take(int count = -1)
            => new TakeDataset(this, count: count);

        public IDatasetV2 batch(int batch_size, bool drop_remainder = false)
            => new BatchDataset(this, batch_size, drop_remainder: drop_remainder);

        public IDatasetV2 prefetch(int buffer_size = -1, int? slack_period = null)
            => new PrefetchDataset(this, buffer_size: buffer_size, slack_period: slack_period);

        public IDatasetV2 repeat(int count = -1)
            => new RepeatDataset(this, count: count);

        public IDatasetV2 shuffle(int buffer_size, int? seed = null, bool reshuffle_each_iteration = true)
            => new ShuffleDataset(this, buffer_size, seed: seed, reshuffle_each_iteration: reshuffle_each_iteration);
        
        public override string ToString()
            => $"{GetType().Name} shapes: ({_structure[0].shape}, {_structure[1].shape}), types: (tf.{_structure[0].dtype.as_numpy_name()}, tf.{_structure[1].dtype.as_numpy_name()})";

        public IEnumerator<(Tensor, Tensor)> GetEnumerator()
        {
            throw new NotImplementedException();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}
