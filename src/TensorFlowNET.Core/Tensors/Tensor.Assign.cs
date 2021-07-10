using System;

namespace Tensorflow
{
    public partial class Tensor
    {
        /// <summary>
        /// Used to keep the original variable when slicing
        /// </summary>
        public ResourceVariable OriginalVar { get; set; }
        public ParsedSliceArgs OriginalVarSlice { get; set; }

        public ResourceVariable assign(Tensor tensor)
        {
            if (tensor.dtype != dtype)
                throw new ArrayTypeMismatchException("");

            if (OriginalVar != null)
            {
                OriginalVar.StridedSliceAssign(tensor, OriginalVarSlice);
                return OriginalVar;
            }
            else
            {
                throw new RuntimeError($"Operation doesn't support. {this.name} is a constant tensor. Make sure to initiate {this.name} from tf.Variable() and declare {this.name} as ResourceVariable or var.");
            }
        }
    }
}
