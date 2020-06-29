using System.Collections.Generic;
using System.Linq;
using System;
using static Tensorflow.OpDef.Types;
using static Tensorflow.Binding;
using Google.Protobuf.WellKnownTypes;
using System.Threading;
using Tensorflow.Util;
using System.Runtime.InteropServices.ComTypes;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class EagerRunner
    {
        int kFastPathExecuteInputStartIndex = 0;

        public Tensor[] TFE_FastPathExecute(Context ctx,
            string device_name,
            string opName,
            string name,
            Action callbacks,
            params object[] args)
        {
            if (ctx == null)
                throw new ValueError("This function does not handle the case of the path where " +
                    "all inputs are not already EagerTensors.");

            int args_size = args.Length;
            var attr_list_sizes = new Dictionary<string, long>();

            FastPathOpExecInfo op_exec_info = new FastPathOpExecInfo()
            {
                ctx = ctx,
                args = args,
                device_name = device_name,
                op_name = opName,
                name = name,
            };

            op_exec_info.run_gradient_callback = HasAccumulatorOrTape();
            op_exec_info.run_post_exec_callbacks = callbacks != null;
            op_exec_info.run_callbacks = op_exec_info.run_gradient_callback || op_exec_info.run_post_exec_callbacks;

            var status = tf.status;
            var op = GetOp(ctx, opName, status);

            var op_def = tf.get_default_graph().GetOpDef(opName);

            // Set non-inferred attrs, including setting defaults if the attr is passed in
            // as None.
            for (int i = kFastPathExecuteInputStartIndex + op_def.InputArg.Count; i < args_size; i += 2)
            {
                var attr_name = args[i].ToString();
                var attr_value = args[i + 1];

                var attr = op_def.Attr.FirstOrDefault(x => x.Name == attr_name);
                if(attr != null)
                {
                    SetOpAttrWithDefaults(ctx, op, attr, attr_name, attr_value, attr_list_sizes, status);
                    status.Check(true);
                }
            }

            var flattened_inputs = args.Take(op_def.InputArg.Count)
                .Select(x => x as Tensor)
                .ToArray();
            var flattened_attrs = args.Skip(op_def.InputArg.Count).ToArray();

            c_api.TFE_OpSetDevice(op, device_name, status.Handle);
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
                if (delta < 0)
                    throw new RuntimeError("Attributes suggest that the size of an output list is less than 0");
                num_retvals += (int)delta;
            }

            var retVals = new IntPtr[num_retvals];
            c_api.TFE_Execute(op, retVals, ref num_retvals, status.Handle);
            status.Check(true);

            var flat_result = retVals.Select(x => new EagerTensor(x)).ToArray();

            if (op_exec_info.run_callbacks)
            {
                if (!RunCallbacks(
                    op_exec_info, 
                    kFastPathExecuteInputStartIndex + op_def.InputArg.Count(),
                    flattened_inputs, flattened_attrs, flat_result))
                {
                    return null;
                }
            }

            return flat_result;
        }

        TFE_Op GetOp(Context ctx, string op_or_function_name, Status status)
        {
            if (thread_local_eager_operation_map.find(ctx, out var op))
                c_api.TFE_OpReset(op, op_or_function_name, ctx.device_name, status.Handle);
            else
            {
                op = c_api.TFE_NewOp(ctx.Handle, op_or_function_name, status.Handle);
                thread_local_eager_operation_map[ctx] = op;
            }
                
            status.Check(true);
            return op;
        }

        static UnorderedMap<Context, TFE_Op> thread_local_eager_operation_map = new UnorderedMap<Context, TFE_Op>();

        bool HasAccumulator()
        {
            //return !GetAccumulatorSet()->empty();
            return false;
        }
        
        bool HasGradientTape()
        {
            return tf.GetTapeSet().Count > 0;
        }

        bool HasAccumulatorOrTape()
        {
            return HasGradientTape() || HasAccumulator();
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
        bool AddInputToOp(object inputs,
            bool add_type_attr,
            ArgDef input_arg,
            IntPtr op,
            Status status)
        {
            IntPtr input_handle;

            // ConvertToTensor();
            switch (inputs)
            {
                case EagerTensor input:
                    input_handle = input.EagerTensorHandle;
                    break;
                case EagerTensor[] input_list:
                    input_handle = input_list[0].EagerTensorHandle;
                    break;
                default:
                    var tensor = tf.convert_to_tensor(inputs);
                    input_handle = (tensor as EagerTensor).EagerTensorHandle;
                    break;
            }

            if (add_type_attr && !string.IsNullOrEmpty(input_arg.TypeAttr))
            {
                var dtype = c_api.TFE_TensorHandleDataType(input_handle);
                c_api.TFE_OpSetAttrType(op, input_arg.TypeAttr, dtype);
            }

            c_api.TFE_OpAddInput(op, input_handle, status.Handle);
            status.Check(true);

            return true;
        }

        public void SetOpAttrs(TFE_Op op, params object[] attrs)
        {
            var status = tf.status;
            var len = attrs.Length;
            for (int i = 0; i < len; i += 2)
            {
                var key = attrs[i].ToString();
                var value = attrs[i + 1];

                byte is_list = 0; 
                var type = c_api.TFE_OpGetAttrType(op, key, ref is_list, status.Handle);
                if (!status.ok()) return;
                if (is_list != 0)
                    SetOpAttrList(tf.context, op, key, value, type, null, status);
                else
                    SetOpAttrScalar(tf.context, op, key, value, type, null, status);
                status.Check(true);
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
        void SetOpAttrWithDefaults(Context ctx, IntPtr op, AttrDef attr, 
            string attr_name, object attr_value,  
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            byte is_list = 0;
            var type = c_api.TFE_OpGetAttrType(op, attr_name, ref is_list, status.Handle);
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

        bool SetOpAttrList(Context ctx, IntPtr op,
            string key, object value, TF_AttrType type,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            return false;
        }

        bool SetOpAttrScalar(Context ctx, IntPtr op, 
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
                    c_api.TFE_OpSetAttrShape(op, key, dims, dims.Length, status.Handle);
                    status.Check(true);
                    break;
                default:
                    throw new NotImplementedException($"SetOpAttrScalar for {type}");
            }

            return true;
        }
    }
}
