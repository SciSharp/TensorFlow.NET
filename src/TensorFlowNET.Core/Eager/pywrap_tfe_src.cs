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
            Action callbacks,
            params object[] args)
        {
            int args_size = args.Length;
            IntPtr op = IntPtr.Zero;
            var attr_list_sizes = new Dictionary<string, long>();
            using (var status = new Status())
            {
                op = c_api.TFE_NewOp(ctx, opName, status);

                var op_def = Graph.TFE_GetOpDef(opName);

                // Set non-inferred attrs, including setting defaults if the attr is passed in
                // as None.
                for (int i = op_def.InputArg.Count; i < args_size; i += 2)
                {
                    var attr_name = args[i].ToString();
                    var attr_value = args[i + 1];

                    foreach(var attr in op_def.Attr)
                    {
                        if(attr_name == attr.Name)
                        {
                            SetOpAttrWithDefaults(ctx, op, attr, attr_name, attr_value, attr_list_sizes, status);
                            break;
                        }
                    }
                }

                c_api.TFE_OpSetDevice(op, device_name, status);

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
                        switch (args[i])
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
                    var delta = 1L;
                    if (!string.IsNullOrEmpty(output_arg.NumberAttr))
                        delta = attr_list_sizes[output_arg.NumberAttr];
                    else if (!string.IsNullOrEmpty(output_arg.TypeListAttr))
                        delta = attr_list_sizes[output_arg.TypeListAttr];
                    if(delta < 0)
                        throw new RuntimeError("Attributes suggest that the size of an output list is less than 0");
                    num_retvals += (int)delta;
                }

                var retVals = new IntPtr[num_retvals];
                c_api.TFE_Execute(op, retVals, ref num_retvals, status);
                status.Check(true);

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

        /// <summary>
        /// This function will set the op attrs required. If an attr has the value of
        /// None, then it will read the AttrDef to get the default value and set that
        /// instead. Any failure in this function will simply fall back to the slow
        /// path.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="op"></param>
        /// <param name="attr"></param>
        /// <param name="attr_name"></param>
        /// <param name="attr_value"></param>
        /// <param name="attr_list_sizes"></param>
        /// <param name="status"></param>
        private static void SetOpAttrWithDefaults(Context ctx, IntPtr op, AttrDef attr, 
            string attr_name, object attr_value,  
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            byte is_list = 0;
            var type = c_api.TFE_OpGetAttrType(op, attr_name, ref is_list, status);
            if (status.Code != TF_Code.TF_OK) return;

            if(attr_value == null)
            {
                if (is_list != 0)
                    ;
                //SetOpAttrListDefault
                else
                    ;
                //SetOpAttrScalarDefault
            }
            else
            {
                if (is_list != 0)
                    ;//  SetOpAttrList
                else
                    SetOpAttrScalar(ctx, op, attr_name, attr_value, type, attr_list_sizes, status);
            }
        }

        private static bool SetOpAttrScalar(Context ctx, IntPtr op, 
            string key, object value, TF_AttrType type,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            switch(type)
            {
                case TF_AttrType.TF_ATTR_STRING:
                    c_api.TFE_OpSetAttrString(op, key, value.ToString(), (uint)value.ToString().Length);
                    break;
                case TF_AttrType.TF_ATTR_TYPE:
                    c_api.TFE_OpSetAttrType(op, key, (TF_DataType)value);
                    break;
                case TF_AttrType.TF_ATTR_BOOL:
                    c_api.TFE_OpSetAttrBool(op, key, Convert.ToBoolean(value));
                    break;
                case TF_AttrType.TF_ATTR_INT:
                    c_api.TFE_OpSetAttrInt(op, key, Convert.ToInt64(value));
                    break;
                default:
                    throw new NotImplementedException("");
            }

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
