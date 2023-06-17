using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    /// <summary>
    /// This is a temp solution, which should be removed after refactoring `Tensors`
    /// </summary>
    [Obsolete]
    public class FakeTensorByTensorArray: Tensor
    {
        public TensorArray TensorArray { get; set; }

        public FakeTensorByTensorArray(TensorArray array)
        {
            TensorArray = array;
        }
    }
}
