using Tensorflow.Framework.Models;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that acts as an identity, and models performance.
    /// </summary>
    public class ModelDataset : UnaryUnchangedStructureDataset
    {
        public ModelDataset(IDatasetV2 input_dataset,
            AutotuneAlgorithm algorithm,
            long cpu_budget,
            long ram_budget) :
            base(input_dataset)
        {
            variant_tensor = ops.model_dataset(input_dataset.variant_tensor,
                output_types,
                output_shapes,
                algorithm,
                cpu_budget,
                ram_budget);
        }
    }
}
