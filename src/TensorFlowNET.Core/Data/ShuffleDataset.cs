using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Randomly shuffles the elements of this dataset.
    /// </summary>
    public class ShuffleDataset : UnaryUnchangedStructureDataset
    {
        Tensor _buffer_size;
        Tensor _seed;
        Tensor _seed2;
        bool _reshuffle_each_iteration;

        public ShuffleDataset(IDatasetV2 input_dataset,
            long buffer_size,
            int? seed = null,
            bool reshuffle_each_iteration = true) :
            base(input_dataset)
        {
            _buffer_size = tf.convert_to_tensor(buffer_size, dtype: TF_DataType.TF_INT64, name: "buffer_size");
            (_seed, _seed2) = random_seed.get_seed_tensor(seed);
            _reshuffle_each_iteration = reshuffle_each_iteration;
            var seed_generator = ops.dummy_seed_generator();
            if (tf.Context.executing_eagerly())
                variant_tensor = ops.shuffle_dataset_v3(input_dataset.variant_tensor, _buffer_size,
                    _seed, _seed2, seed_generator,
                    output_types, output_shapes,
                    reshuffle_each_iteration: _reshuffle_each_iteration);
            else
                throw new NotImplementedException("");
        }
    }
}
