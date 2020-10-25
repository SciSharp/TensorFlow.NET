using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Functions;

namespace Tensorflow
{
    /// <summary>
    /// 
    /// </summary>
    public class FlatMapDataset : UnaryDataset
    {
        public FlatMapDataset(IDatasetV2 input_dataset,
            Func<Tensor, IDatasetV2> map_func) : base(input_dataset)
        {
            var func = new ConcreteFunction(map_func, input_dataset.element_spec[0].dtype);

            variant_tensor = ops.flat_map_dataset(input_dataset.variant_tensor,
                func,
                output_types,
                output_shapes);
        }
    }
}
