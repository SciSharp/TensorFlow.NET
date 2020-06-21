using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    public class EagerTensorPass : PointerInputs<EagerTensor>
    {
        public EagerTensorPass(params EagerTensor[] tensors)
        {
            data = tensors;
        }

        public static EagerTensorPass Create(int count = 1)
            => new EagerTensorPass(Enumerable.Range(0, count).Select(x => new EagerTensor()).ToArray());

        public static EagerTensorPass From(params object[] objects)
            => new EagerTensorPass(objects.Select(x => ops.convert_to_tensor(x) as EagerTensor).ToArray());
    }
}
