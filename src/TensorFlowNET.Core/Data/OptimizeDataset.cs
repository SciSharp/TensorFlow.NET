using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that acts as an identity, and applies optimizations.
    /// </summary>
    public class OptimizeDataset : UnaryUnchangedStructureDataset
    {
        public OptimizeDataset(IDatasetV2 dataset,
            string[] optimizations = null,
            string[] optimization_configs = null) :
            base(dataset)
        {
            if (optimizations == null)
                optimizations = new string[0];
            if (optimization_configs == null)
                optimization_configs = new string[0];

            var _optimizations = tf.convert_to_tensor(optimizations, dtype: TF_DataType.TF_STRING, name: "optimizations");
            variant_tensor = ops.optimize_dataset(
                _input_dataset.variant_tensor,
                _optimizations,
                output_types,
                output_shapes,
                optimization_configs: optimization_configs);
        }
    }
}
