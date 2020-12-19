using System;
using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    /// <summary>
    /// Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy.
    /// </summary>
    public class TensorLikeDataAdapter : DataAdapter, IDataAdapter
    {
        int _size;
        int _batch_size;
        int num_samples;
        int num_full_batches;

        public TensorLikeDataAdapter(DataAdapterArgs args)
        {
            this.args = args;
            _process_tensorlike();
            num_samples = args.X.shape[0];
            var batch_size = args.BatchSize == -1 ? 32 : args.BatchSize;
            _batch_size = batch_size;
            _size = Convert.ToInt32(Math.Ceiling(num_samples / (batch_size + 0f)));
            num_full_batches = num_samples / batch_size;
            var _partial_batch_size = num_samples % batch_size;

            var indices_dataset = tf.data.Dataset.range(1);
            indices_dataset = indices_dataset.repeat();
            indices_dataset = indices_dataset.map(permutation).prefetch(1);
            indices_dataset = indices_dataset.flat_map(slice_batch_indices);
            dataset = slice_inputs(indices_dataset, args.X, args.Y);
        }

        Tensor permutation(Tensor tensor)
        {
            var indices = math_ops.range(num_samples, dtype: dtypes.int64);
            if (args.Shuffle)
                indices = random_ops.random_shuffle(indices);
            return indices;
        }

        /// <summary>
        /// Convert a Tensor of indices into a dataset of batched indices.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        IDatasetV2 slice_batch_indices(Tensor indices)
        {
            var num_in_full_batch = num_full_batches * _batch_size;
            var first_k_indices = array_ops.slice(indices, new int[] { 0 }, new int[] { num_in_full_batch });
            first_k_indices = array_ops.reshape(first_k_indices, new int[] { num_full_batches, _batch_size });
            var flat_dataset = tf.data.Dataset.from_tensor_slices(first_k_indices);
            return flat_dataset;
        }

        IDatasetV2 slice_inputs(IDatasetV2 indices_dataset, Tensor x, Tensor y)
        {
            var dataset2 = tf.data.Dataset.from_tensor(x, y).repeat();
            var dataset = tf.data.Dataset.zip(indices_dataset, dataset2);

            dataset = dataset.map((batch, data) =>
            {
                var x = gen_array_ops.gather_v2(data.Item1, batch, 0);
                var y = gen_array_ops.gather_v2(data.Item2, batch, 0);
                return (x, y);
            });

            dataset = dataset.with_options(new DatasetOptions { });

            return dataset;
        }

        public override int GetSize()
            => _size;

        void _process_tensorlike()
        {
        }
    }
}
