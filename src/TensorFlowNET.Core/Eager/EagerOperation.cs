using System;

namespace Tensorflow.Eager
{
    public class EagerOperation : Operation
    {
        public string Name { get; set; }
        public new int NumInputs;
        public IntPtr[] InputHandles { get; set; }
        public Tensor[] Inputs { get; set; }
        public new int NumOutputs;
        public IntPtr[] OutputHandles { get; set; }
        public Tensor[] Outputs { get; set; }
        public long[] SkipInputIndices { get; set; }
        public object[] Attrs { get; set; }

        public EagerOperation() : base(IntPtr.Zero)
        {

        }

        public override InputList inputs
        {
            get
            {
                if (_inputs_val == null)
                {
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

        public override object get_attr(string attr_name)
        {
            // var attrType = c_api.TFE_OpNameGetAttrType(tf.Context.Handle, Name, attr_name, ref isList, tf.Status.Handle);
            for (int i = 0; i < Attrs.Length; i = i + 2)
            {
                if (Attrs[i].Equals(attr_name))
                    return Attrs[i + 1];
            }

            return null;
        }

        public override string ToString()
            => $"tf.EagerOperation {Name}";
    }
}
