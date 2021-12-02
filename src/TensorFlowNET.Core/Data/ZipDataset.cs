using System.Collections.Generic;
using System.Linq;
using Tensorflow.Framework.Models;

namespace Tensorflow
{
    public class ZipDataset : DatasetV2
    {
        // keep all dataset references
        IDatasetV2[] _inputs;
        public ZipDataset(params IDatasetV2[] ds)
        {
            _inputs = ds;
            var input_datasets = ds.Select(x => x.variant_tensor).ToArray();
            var _structure = new List<TensorSpec>();
            foreach (var dataset in ds)
                _structure.AddRange(dataset.structure);
            structure = _structure.ToArray();
            variant_tensor = ops.zip_dataset(input_datasets, output_types, output_shapes);
        }
    }
}
