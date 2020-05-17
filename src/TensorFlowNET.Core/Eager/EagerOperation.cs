using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class EagerOperation : Operation
    {
        public int NumInputs;
        public Tensor[] Inputs { get; set; }
        public int NumOutputs;
        public Tensor[] Outputs { get; set; }
        public int[] SkipInputIndices { get; set; }

        public EagerOperation() : base(IntPtr.Zero) { }

        public override InputList inputs
        {
            get
            {
                if (_inputs_val == null)
                {
                    var retval = new Tensor[NumInputs];

                    for (int i = 0; i < NumInputs; i++)
                    {

                    }

                    _inputs_val = new InputList(Inputs);
                }

                return _inputs_val;
            }
        }

        public override Tensor[] outputs
        {
            get
            {
                if (_outputs == null)
                {
                    _outputs = Outputs;
                }

                return _outputs;
            }
        }
    }
}
