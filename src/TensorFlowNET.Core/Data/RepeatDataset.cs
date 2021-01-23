namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that repeats its input several times.
    /// </summary>
    public class RepeatDataset : UnaryUnchangedStructureDataset
    {
        public RepeatDataset(IDatasetV2 input_dataset, int count = -1) :
            base(input_dataset)
        {
            var count_tensor = constant_op.constant(count, dtype: TF_DataType.TF_INT64, name: "count");
            variant_tensor = ops.repeat_dataset(input_dataset.variant_tensor,
                count_tensor,
                input_dataset.output_types,
                input_dataset.output_shapes);
        }
    }
}
