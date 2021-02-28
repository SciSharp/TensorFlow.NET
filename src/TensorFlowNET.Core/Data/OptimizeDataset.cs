using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that acts as an identity, and applies optimizations.
    /// </summary>
    public class OptimizeDataset : UnaryUnchangedStructureDataset
    {
        public OptimizeDataset(IDatasetV2 dataset,
            string[] optimizations_enabled = null,
            string[] optimizations_disabled = null,
            string[] optimizations_default = null,
            string[] optimization_configs = null) :
            base(dataset)
        {
            if (optimizations_enabled == null)
                optimizations_enabled = new string[0];
            if (optimizations_disabled == null)
                optimizations_disabled = new string[0];
            if (optimizations_default == null)
                optimizations_default = new string[0];
            if (optimization_configs == null)
                optimization_configs = new string[0];

            var _optimizations_enabled = tf.convert_to_tensor(optimizations_enabled, dtype: TF_DataType.TF_STRING, name: "optimizations_enabled");
            var _optimizations_disabled = tf.convert_to_tensor(optimizations_disabled, dtype: TF_DataType.TF_STRING, name: "optimizations_disabled");
            var _optimizations_default = tf.convert_to_tensor(optimizations_default, dtype: TF_DataType.TF_STRING, name: "optimizations_default");

            variant_tensor = ops.optimize_dataset_v2(
                _input_dataset.variant_tensor,
                _optimizations_enabled,
                _optimizations_disabled,
                _optimizations_default,
                output_types,
                output_shapes,
                optimization_configs: optimization_configs);
        }
    }
}
