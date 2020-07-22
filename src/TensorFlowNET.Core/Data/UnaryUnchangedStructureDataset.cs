using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Represents a unary dataset with the same input and output structure.
    /// </summary>
    public class UnaryUnchangedStructureDataset : UnaryDataset
    {
        public UnaryUnchangedStructureDataset(IDatasetV2 input_dataset) :
            base(input_dataset)
        {
            
        }
    }
}
