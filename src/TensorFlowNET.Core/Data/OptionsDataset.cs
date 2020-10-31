using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// An identity `Dataset` that stores options.
    /// </summary>
    public class OptionsDataset : UnaryUnchangedStructureDataset
    {
        DatasetOptions options;

        public OptionsDataset(IDatasetV2 input_dataset, DatasetOptions options) 
            : base(input_dataset)
        {
            this.options = options;
            variant_tensor = input_dataset.variant_tensor;
        }
    }
}
