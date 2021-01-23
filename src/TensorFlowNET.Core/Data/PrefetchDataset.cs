using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Creates a `Dataset` that prefetches elements from this dataset.
    /// </summary>
    public class PrefetchDataset : UnaryUnchangedStructureDataset
    {
        public PrefetchDataset(IDatasetV2 input_dataset,
            long buffer_size = -1,
            int? slack_period = null) :
            base(input_dataset)
        {
            var buffer_size_tensor = tf.convert_to_tensor(buffer_size, dtype: TF_DataType.TF_INT64, name: "buffer_size");

            variant_tensor = ops.prefetch_dataset(input_dataset.variant_tensor,
                buffer_size_tensor,
                input_dataset.output_types,
                input_dataset.output_shapes,
                slack_period: slack_period);
        }
    }
}
