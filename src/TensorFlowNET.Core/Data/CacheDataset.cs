using static Tensorflow.Binding;

namespace Tensorflow
{
    public class CacheDataset : UnaryUnchangedStructureDataset
    {
        Tensor _filename;
        public CacheDataset(IDatasetV2 input_dataset,
            string filename = "") :
            base(input_dataset)
        {
            _filename = tf.convert_to_tensor(filename, dtype: TF_DataType.TF_STRING, name: "filename");
            variant_tensor = ops.cache_dataset_v2(input_dataset.variant_tensor,
                _filename,
                ops.dummy_memory_cache(),
                output_types,
                output_shapes);
        }
    }
}
