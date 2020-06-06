using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public class EagerOperation : Operation
    {
        static Dictionary<string, OpDef> op_dict;
        public string Name { get; set; }
        public new int NumInputs;
        public IntPtr[] InputHandles { get; set; }
        public Tensor[] Inputs { get; set; }
        public new int NumOutputs;
        public IntPtr[] OutputHandles { get; set; }
        public Tensor[] Outputs { get; set; }
        public BindingArray SkipInputIndicesArray { get; set; }
        public unsafe int[] SkipInputIndices => SkipInputIndicesArray.Data.Select(x => *(int*) x).ToArray();
        public BindingArray AttrsArray { get; set; }

        public EagerOperation() : base(IntPtr.Zero)
        {
            if (op_dict == null)
                op_dict = op_def_registry.get_registered_ops();
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
            object value = null;
            byte isList = 0;
            using var status = new Status();
            var attrType = c_api.TFE_OpNameGetAttrType(tf.context, Name, attr_name, ref isList, status);
            switch (attrType)
            {
                case TF_AttrType.TF_ATTR_BOOL:
                    value = get_attr_bool(attr_name);
                    break;
                default:
                    break;
            }

            return value;
        }

        public bool get_attr_bool(string attr_name) 
        {
            for (int i = 0; i < AttrsArray.Data.Length; i = i + 2)
            {
                //  var a = *(char*)AttrsArray.Data[i];
                if (op_dict[Name].Attr[i].Name == attr_name)
                    return (int)AttrsArray.Data[i + 1] == 1;
            }

            throw new ValueError($"Can't find attr: {attr_name}");
        }

        public override string ToString()
            => $"tf.EagerOperation {Name}";
    }
}
