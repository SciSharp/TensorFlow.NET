using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework.Models;

namespace Tensorflow
{
    /// <summary>
    /// Abstract class representing a dataset with one input.
    /// </summary>
    public class UnaryDataset : DatasetV2
    {
        protected IDatasetV2 _input_dataset;

        public UnaryDataset(IDatasetV2 input_dataset)
        {
            _input_dataset = input_dataset;
            structure = input_dataset.structure;
        }
    }
}
