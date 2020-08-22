using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow.Data
{
    public class RangeDataset : DatasetSource
    {
        Tensor start;
        Tensor step;
        Tensor stop;

        public RangeDataset(int stop, 
            int start = 0, 
            int step = 1,
            TF_DataType output_type = TF_DataType.TF_INT64)
        {
            this.start = tf.convert_to_tensor((long)start);
            this.step = tf.convert_to_tensor((long)step);
            this.stop = tf.convert_to_tensor((long)stop);

            structure = new TensorSpec[] { new TensorSpec(new int[0], dtype: output_type) };
            variant_tensor = ops.range_dataset(this.start, this.stop, this.step, output_types, output_shapes);
        }
    }
}
