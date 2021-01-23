using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow.Data
{
    public class RangeDataset : DatasetSource
    {
        public RangeDataset(int stop,
            int start = 0,
            int step = 1,
            TF_DataType output_type = TF_DataType.TF_INT64)
        {
            var start_tensor = tf.convert_to_tensor((long)start);
            var step_tensor = tf.convert_to_tensor((long)step);
            var stop_tensor = tf.convert_to_tensor((long)stop);

            structure = new TensorSpec[] { new TensorSpec(new int[0], dtype: output_type) };
            variant_tensor = ops.range_dataset(start_tensor, stop_tensor, step_tensor, output_types, output_shapes);
        }
    }
}
