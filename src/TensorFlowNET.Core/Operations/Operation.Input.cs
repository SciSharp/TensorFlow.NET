using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Operation
    {
        public TF_Output Input(int index) => c_api.TF_OperationInput(new TF_Input(_handle, index));
        public TF_DataType InputType(int index) => c_api.TF_OperationInputType(new TF_Input(_handle, index));
        public int InputListLength(string name) => c_api.TF_OperationInputListLength(_handle, name, status);
        public int NumInputs => c_api.TF_OperationNumInputs(_handle);
        private TF_DataType[] _input_types => _inputs._inputs.Select(x => x.dtype).ToArray();

        private InputList _inputs;
        public InputList inputs
        {
            get
            {
                if (_inputs == null)
                {
                    var retval = new Tensor[NumInputs];

                    for (int i = 0; i < NumInputs; i++)
                    {
                        var tf_outpus = Input(i);
                        var op = new Operation(tf_outpus.oper);
                        retval[i] = op.outputs[tf_outpus.index];
                    }

                    _inputs = new InputList(retval);
                }

                return _inputs;
            }
        }

        public int NumControlInputs => c_api.TF_OperationNumControlInputs(_handle);

        public Operation[] control_inputs
        {
            get
            {
                return GetControlInputs();
            }
        }

        public unsafe Operation[] GetControlInputs()
        {
            var control_inputs = new Operation[NumControlInputs];

            if (NumControlInputs > 0)
            {
                IntPtr control_input_handle = Marshal.AllocHGlobal(Marshal.SizeOf<IntPtr>() * NumControlInputs);
                c_api.TF_OperationGetControlInputs(_handle, control_input_handle, NumControlInputs);
                for (int i = 0; i < NumControlInputs; i++)
                {
                    var handle = control_input_handle + Marshal.SizeOf<IntPtr>() * i;
                    control_inputs[i] = new Operation(*(IntPtr*)handle);
                }
            }

            return control_inputs;
        }
    }
}
