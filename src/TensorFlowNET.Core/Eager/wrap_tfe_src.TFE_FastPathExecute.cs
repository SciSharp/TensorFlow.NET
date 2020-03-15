using System.Collections.Generic;
using System.Linq;
using System;
using static Tensorflow.OpDef.Types;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class wrap_tfe_src
    {
        static int kFastPathExecuteInputStartIndex = 0;
        public static EagerTensor TFE_FastPathExecute(Context ctx, 
            string device_name, 
            string opName, 
            string name, 
            Action callbacks,
            params object[] args)
        {
            int args_size = args.Length;
            var attr_list_sizes = new Dictionary<string, long>();
            using (var status = new Status())
            {
                var op = c_api.TFE_NewOp(ctx, opName, status);

                var op_def = Graph.TFE_GetOpDef(opName);

                // Set non-inferred attrs, including setting defaults if the attr is passed in
                // as None.
                for (int i = kFastPathExecuteInputStartIndex + op_def.InputArg.Count; i < args_size; i += 2)
                {
                    var attr_name = args[i].ToString();
                    var attr_value = args[i + 1];

                    foreach(var attr in op_def.Attr)
                    {
                        if(attr_name == attr.Name)
                        {
                            SetOpAttrWithDefaults(ctx, op, attr, attr_name, attr_value, attr_list_sizes, status);
                            status.Check(true);
                            break;
                        }
                    }
                }

                c_api.TFE_OpSetDevice(op, device_name, status);
                status.Check(true);

                // Add inferred attrs and inputs.
                for (int i = 0; i < op_def.InputArg.Count; i++)
                {
                    var input_arg = op_def.InputArg[i];
                    if (!string.IsNullOrEmpty(input_arg.NumberAttr))
                    {
                        int len = (args[kFastPathExecuteInputStartIndex + i] as object[]).Length;
                        c_api.TFE_OpSetAttrInt(op, input_arg.NumberAttr, len);
                        attr_list_sizes[input_arg.NumberAttr] = len;

                        if (len > 0)
                        {
                            var fast_input_array = (object[])args[i];
                            // First item adds the type attr.
                            if (!AddInputToOp(fast_input_array[i], true, input_arg, op, status))
                                return null;

                            for (var j = 1; j < len; j++)
                            {
                                // Since the list is homogeneous, we don't need to re-add the attr.
                                if (!AddInputToOp(fast_input_array[j], false, input_arg, op, status))
                                    return null;
                            }
                        }
                    }
                    else if (!string.IsNullOrEmpty(input_arg.TypeListAttr))
                    {

                    }
                    else
                    {
                        // The item is a single item.
                        AddInputToOp(args[i], true, input_arg, op, status);
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
        private static bool AddInputToOp(object inputs,
            bool add_type_attr,
            ArgDef input_arg,
            IntPtr op, 
            Status status)
        {
            TFE_TensorHandle input_handle;

            switch (inputs)
            {
                case Tensor input:
                    input_handle = c_api.TFE_NewTensorHandle(input, status);
                    break;
                case Tensor[] input_list:
                    input_handle = c_api.TFE_NewTensorHandle(input_list[0], status);
                    break;
                default:
                    throw new NotImplementedException("");
            }


            if(add_type_attr && !string.IsNullOrEmpty(input_arg.TypeAttr))
            {
                var dtype = c_api.TFE_TensorHandleDataType(input_handle);
                c_api.TFE_OpSetAttrType(op, input_arg.TypeAttr, dtype);
            }

            c_api.TFE_OpAddInput(op, input_handle, status);
            status.Check(true);

            return true;
        }

        private static void SetOpAttrs(Context ctx, TFE_Op op, object[] attrs, int start_index, Status out_status)
        {
            var len = attrs.Length;
            for (int i = 0; i < len; i += 2)
            {
                var key = attrs[start_index + i].ToString();
                var value = attrs[start_index + i + 1];

                byte is_list = 0;
                var type = c_api.TFE_OpGetAttrType(op, key, ref is_list, out_status);
                if (!out_status.ok()) return;
                if (is_list != 0)
                    SetOpAttrList(ctx, op, key, value, type, null, out_status);
                else
                    SetOpAttrScalar(ctx, op, key, value, type, null, out_status);
                out_status.Check(true);
            }
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

        private static bool SetOpAttrList(Context ctx, IntPtr op,
            string key, object value, TF_AttrType type,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            return false;
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
                case TF_AttrType.TF_ATTR_SHAPE:
                    var dims = (value as int[]).Select(x => (long)x).ToArray();
                    c_api.TFE_OpSetAttrShape(op, key, dims, dims.Length, status);
                    status.Check(true);
                    break;
                default:
                    throw new NotImplementedException($"SetOpAttrScalar for {type}");
            }

            return true;
        }
    }
}
