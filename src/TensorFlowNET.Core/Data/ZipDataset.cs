using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class ZipDataset : DatasetV2
    {
        dataset_ops ops = new dataset_ops();
        public ZipDataset(params IDatasetV2[] ds)
        {
            var input_datasets = ds.Select(x => x.variant_tensor).ToArray();
            structure = ds.Select(x => x.structure[0]).ToArray();
            variant_tensor = ops.zip_dataset(input_datasets, output_types, output_shapes);
        }
    }
}
