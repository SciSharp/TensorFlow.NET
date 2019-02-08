using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class BulkSaverBuilder : BaseSaverBuilder, ISaverBuilder
    {
        public BulkSaverBuilder(int write_version = 2) : base(write_version)
        {

        }
    }
}
