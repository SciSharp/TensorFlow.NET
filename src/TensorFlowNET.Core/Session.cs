using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Session : BaseSession
    {
        public override byte[] run(Tensor fetches)
        {
            var ret = base.run(fetches);

            return ret;
        }
    }
}
