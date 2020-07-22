using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework.Models;

namespace Tensorflow
{
    public class DatasetSource : DatasetV2
    {
        protected Tensor[] _tensors;

        public DatasetSource()
        {

        }
    }
}
