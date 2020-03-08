using System.Collections.Generic;
using System.Linq;
using System;
using static Tensorflow.OpDef.Types;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public class pywrap_tfe_src
    {
        public static EagerTensor TFE_Py_FastPathExecute(Context ctx, 
            string device_name, 
            string opName, 
            string name, 
            params object[] inputs)
        {
            IntPtr op = IntPtr.Zero;
            var attr_list_sizes = new Dictionary<string, int>();
            using (var status = new Status())
            {
                op = c_api.TFE_NewOp(ctx, opName, status);

                var op_def = Graph.TFE_GetOpDef(opName);

                // SetOpAttrWithDefaults
                c_api.TFE_OpSetDevice(op, "", status);

                for (int i = 0; i < op_def.InputArg.Count; i++)
                {
                    var input_arg = op_def.InputArg[i];
                    if (!string.IsNullOrEmpty(input_arg.NumberAttr))
                    {
                        c_api.TFE_OpSetAttrInt(op, input_arg.NumberAttr, 0);
                        attr_list_sizes[input_arg.NumberAttr] = 0;
                    }
                    else if (!string.IsNullOrEmpty(input_arg.TypeListAttr))
                    {

                    }
                    else
                    {
                        // The item is a single item.
                        switch (inputs[i])
                        {
                            case Tensor inputTensor:
                                AddInputToOp(inputTensor, true, input_arg, op, status);
                                break;
                            default:
                                throw new NotImplementedException("");
                        }
                    }
                }

                int num_retvals = 0;
                for (int i = 0; i < op_def.OutputArg.Count; i++)
                {
                    var output_arg = op_def.OutputArg[i];
                    var delta = 1;
                    if (!string.IsNullOrEmpty(output_arg.NumberAttr))
                        delta = attr_list_sizes[output_arg.NumberAttr];
                    else if (!string.IsNullOrEmpty(output_arg.TypeListAttr))
                        delta = attr_list_sizes[output_arg.TypeListAttr];
                    if(delta < 0)
                        throw new RuntimeError("Attributes suggest that the size of an output list is less than 0");
                    num_retvals += delta;
                }

                var retVals = new IntPtr[num_retvals];
                c_api.TFE_Execute(op, retVals, ref num_retvals, status);

                var t = c_api.TFE_TensorHandleResolve(retVals[0], status);
                status.Check(true);

                return new EagerTensor(t);
            }
        }

        /// <summary>
        /// Adds input and type attr to the op, and to the list of flattened
        /// inputs/attrs.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="add_type_attr"></param>
        /// <param name="input_arg"></param>
        /// <param name="op"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        private static bool AddInputToOp(Tensor input,
            bool add_type_attr,
            ArgDef input_arg,
            IntPtr op, 
            Status status)
        {
            var input_handle = c_api.TFE_NewTensorHandle(input, status);

            if(add_type_attr && !string.IsNullOrEmpty(input_arg.TypeAttr))
            {
                var dtype = c_api.TFE_TensorHandleDataType(input_handle);
                c_api.TFE_OpSetAttrType(op, input_arg.TypeAttr, dtype);
            }

            c_api.TFE_OpAddInput(op, input_handle, status);
            status.Check(true);
            return true;
        }

        public static void RecordGradient(string op_name, Tensor[] inputs, Dictionary<string, object> attrs, Tensor[] results, string name = null)
        {
            var input_ids = inputs.Select(x => x.Id).ToArray();
            var input_dtypes = inputs.Select(x => x.dtype).ToArray();

            bool should_record = false;
            foreach (var input_dtype in input_dtypes)
            {
                if (Tape.IsDtypeTrainable(input_dtype.as_datatype_enum()))
                {
                    should_record = true;
                    break;
                }
            }
            if (!should_record) return;

            var op_outputs = results;
            var op_inputs = inputs;
        }
    }
}
