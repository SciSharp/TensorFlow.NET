using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public struct TF_AttrMetadata
    {
        public char is_list;
        public long list_size;
        public TF_AttrType type;
        public long total_size;
    }
}
