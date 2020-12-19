using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    public class DatasetAdapter : DataAdapter, IDataAdapter
    {
        public DatasetAdapter(DataAdapterArgs args)
        {
            this.args = args;
            dataset = args.Dataset;
        }

        public override int GetSize()
            => -1;
    }
}
