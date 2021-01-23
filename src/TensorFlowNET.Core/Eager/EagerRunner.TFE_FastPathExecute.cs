using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Tensorflow.Contexts;
using Tensorflow.Functions;
using Tensorflow.Util;
using static Tensorflow.Binding;
using static Tensorflow.OpDef.Types;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class EagerRunner
    {
        int kFastPathExecuteInputStartIndex = 0;
        UnorderedMap<Context, SafeOpHandle> thread_local_eager_operation_map = new UnorderedMap<Context, SafeOpHandle>();

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

            var status = tf.Status;
            using var op = GetOp(ctx, opName, status);

            var op_def = tf.get_default_graph().GetOpDef(opName);

            var flattened_attrs = new List<object>(op_def.Attr.Count * 2);
            var flattened_inputs = new List<Tensor>(op_def.InputArg.Count);

            // Set non-inferred attrs, including setting defaults if the attr is passed in
            // as None.
            for (int i = kFastPathExecuteInputStartIndex + op_def.InputArg.Count; i < args_size; i += 2)
            {
                var attr_name = args[i].ToString();
                var attr_value = args[i + 1];

                var attr = op_def.Attr.FirstOrDefault(x => x.Name == attr_name);
                if (attr != null)
                {
                    flattened_attrs.Add(attr_name);
                    flattened_attrs.Add(attr_value);

                    SetOpAttrWithDefaults(ctx, op, attr, attr_name, attr_value, attr_list_sizes, status);
                    status.Check(true);
                }
            }

            c_api.TFE_OpSetDevice(op, device_name, status.Handle);
            status.Check(true);

            // Add inferred attrs and inputs.
            for (int i = 0; i < op_def.InputArg.Count; i++)
            {
                var input = args[kFastPathExecuteInputStartIndex + i];
                var input_arg = op_def.InputArg[i];
                if (!string.IsNullOrEmpty(input_arg.NumberAttr))
                {
                    int len = (input as object[]).Length;
                    c_api.TFE_OpSetAttrInt(op, input_arg.NumberAttr, len);
                    if (op_exec_info.run_callbacks)
                    {
                        flattened_attrs.Add(input_arg.NumberAttr);
                        flattened_attrs.Add(len);
                    }
                    attr_list_sizes[input_arg.NumberAttr] = len;

                    if (len > 0)
                    {
                        var fast_input_array = (object[])args[i];
                        // First item adds the type attr.
                        if (!AddInputToOp(fast_input_array[i], true, input_arg, flattened_attrs, flattened_inputs, op, status))
                            return null;

                        for (var j = 1; j < len; j++)
                        {
                            // Since the list is homogeneous, we don't need to re-add the attr.
                            if (!AddInputToOp(fast_input_array[j], false, input_arg, flattened_attrs, flattened_inputs, op, status))
                                return null;
                        }
                    }
                }
                else if (!string.IsNullOrEmpty(input_arg.TypeListAttr))
                {
                    var attr_name = input_arg.TypeListAttr;
                    var fast_input_array = input as object[];
                    var len = fast_input_array.Length;
                    var attr_values = new TF_DataType[len];

                    for (var j = 0; j < len; j++)
                    {
                        var eager_tensor = ops.convert_to_tensor(fast_input_array[j]);
                        attr_values[j] = eager_tensor.dtype;

                        c_api.TFE_OpAddInput(op, eager_tensor.EagerTensorHandle, status.Handle);

                        if (op_exec_info.run_callbacks)
                        {
                            flattened_inputs.Add(eager_tensor);
                        }
                    }

                    if (op_exec_info.run_callbacks)
                    {
                        flattened_attrs.Add(attr_name);
                        flattened_attrs.Add(attr_values);
                    }
                    c_api.TFE_OpSetAttrTypeList(op, attr_name, attr_values, attr_values.Length);
                    attr_list_sizes[attr_name] = len;
                }
                else
                {
                    // The item is a single item.
                    AddInputToOp(args[i], true, input_arg, flattened_attrs, flattened_inputs, op, status);
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

            var retVals = new SafeTensorHandleHandle[num_retvals];
            c_api.TFE_Execute(op, retVals, out num_retvals, status.Handle);
            status.Check(true);

            var flat_result = retVals.Select(x => new EagerTensor(x)).ToArray();


            if (op_exec_info.run_callbacks)
            {
                RunCallbacks(op_exec_info,
                    kFastPathExecuteInputStartIndex + op_def.InputArg.Count(),
                    flattened_inputs.ToArray(), flattened_attrs.ToArray(), flat_result);
            }

            return flat_result;
        }

        SafeOpHandle GetOp(Context ctx, string op_or_function_name, Status status)
        {
            /*if (thread_local_eager_operation_map.find(ctx, out var op))
                c_api.TFE_OpReset(op, op_or_function_name, ctx.DeviceName, status.Handle);
            else
            {
                op = c_api.TFE_NewOp(ctx.Handle, op_or_function_name, status.Handle);
                thread_local_eager_operation_map[ctx] = op;
            }

            status.Check(true);
            return op;*/
            var op = c_api.TFE_NewOp(ctx.Handle, op_or_function_name, status.Handle);
            status.Check(true);
            return op;
        }

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
            List<object> flattened_attrs,
            List<Tensor> flattened_inputs,
            SafeOpHandle op,
            Status status)
        {
            var tensor = tf.convert_to_tensor(inputs);
            flattened_inputs.Add(tensor);

            if (add_type_attr && !string.IsNullOrEmpty(input_arg.TypeAttr))
            {
                var dtype = c_api.TFE_TensorHandleDataType(tensor.EagerTensorHandle);
                c_api.TFE_OpSetAttrType(op, input_arg.TypeAttr, dtype);
                flattened_attrs.Add(input_arg.TypeAttr);
                flattened_attrs.Add(dtype);
            }

            c_api.TFE_OpAddInput(op, tensor.EagerTensorHandle, status.Handle);
            status.Check(true);

            return true;
        }

        public void SetOpAttrs(SafeOpHandle op, params object[] attrs)
        {
            var status = tf.Status;
            var len = attrs.Length;
            for (int i = 0; i < len; i += 2)
            {
                var key = attrs[i].ToString();
                var value = attrs[i + 1];

                byte is_list = 0;
                var type = c_api.TFE_OpGetAttrType(op, key, ref is_list, status.Handle);
                if (!status.ok()) return;
                if (is_list != 0)
                    SetOpAttrList(tf.Context, op, key, value as object[], type, null, status);
                else
                    SetOpAttrScalar(tf.Context, op, key, value, type, null, status);
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
        void SetOpAttrWithDefaults(Context ctx, SafeOpHandle op, AttrDef attr,
            string attr_name, object attr_value,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            byte is_list = 0;
            var type = c_api.TFE_OpGetAttrType(op, attr_name, ref is_list, status.Handle);
            if (status.Code != TF_Code.TF_OK) return;

            if (attr_value == null)
            {
                if (is_list != 0)
#pragma warning disable CS0642 // Possible mistaken empty statement
                    ;
#pragma warning restore CS0642 // Possible mistaken empty statement
                //SetOpAttrListDefault
                else
#pragma warning disable CS0642 // Possible mistaken empty statement
                    ;
#pragma warning restore CS0642 // Possible mistaken empty statement
                //SetOpAttrScalarDefault
            }
            else
            {
                if (is_list != 0)
                    SetOpAttrList(ctx, op, attr_name, attr_value, type, attr_list_sizes, status);
                else
                    SetOpAttrScalar(ctx, op, attr_name, attr_value, type, attr_list_sizes, status);
            }
        }

        bool SetOpAttrList(Context ctx, SafeOpHandle op,
            string key, object values, TF_AttrType type,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            if (type == TF_AttrType.TF_ATTR_STRING && values is string[] values3)
            {
                c_api.TFE_OpSetAttrStringList(op, key, new IntPtr[0], values3.Select(x => x.Length).ToArray(), values3.Length);
                attr_list_sizes[key] = values3.Length;
            }
            else if (type == TF_AttrType.TF_ATTR_SHAPE && values is TensorShape[] values1)
            {
                // Make one pass through the input counting the total number of
                // dims across all the input lists.
                var num_values = values1.Length;
                attr_list_sizes[key] = num_values;
                var dims = new IntPtr[num_values];
                var num_dims = values1.Select(x => x.ndim).ToArray();

                for (int i = 0; i < num_values; ++i)
                {
                    dims[i] = Marshal.AllocHGlobal(sizeof(long) * values1[i].ndim);
                    tf.memcpy(dims[i], values1[i].dims.Select(x => (long)x).ToArray(), values1[i].ndim * sizeof(long));
                }

                c_api.TFE_OpSetAttrShapeList(op, key, dims, num_dims, num_values, status.Handle);
                Array.ForEach(dims, x => Marshal.FreeHGlobal(x));
            }
            else if (type == TF_AttrType.TF_ATTR_TYPE && values is TF_DataType[] values2)
            {
                c_api.TFE_OpSetAttrTypeList(op, key, values2, values2.Length);
                attr_list_sizes[key] = values2.Length;
            }
            else if (type == TF_AttrType.TF_ATTR_INT && values is int[] values4)
            {
                c_api.TFE_OpSetAttrIntList(op, key, values4.Select(x => Convert.ToInt64(x)).ToArray(), values4.Length);
                attr_list_sizes[key] = values4.Length;
            }
            else
            {
                throw new NotImplementedException("");
            }

            return true;
        }

        bool SetOpAttrScalar(Context ctx, SafeOpHandle op,
            string key, object value, TF_AttrType type,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            switch (type)
            {
                case TF_AttrType.TF_ATTR_STRING:
                    c_api.TFE_OpSetAttrString(op, key, value.ToString(), (ulong)value.ToString().Length);
                    break;
                case TF_AttrType.TF_ATTR_TYPE:
                    c_api.TFE_OpSetAttrType(op, key, (TF_DataType)value);
                    break;
                case TF_AttrType.TF_ATTR_BOOL:
                    c_api.TFE_OpSetAttrBool(op, key, Convert.ToBoolean(value));
                    break;
                case TF_AttrType.TF_ATTR_INT:
                    var size = Convert.ToInt64(value);
                    c_api.TFE_OpSetAttrInt(op, key, size);
                    if (attr_list_sizes != null)
                        attr_list_sizes[key] = size;
                    break;
                case TF_AttrType.TF_ATTR_FLOAT:
                    c_api.TFE_OpSetAttrFloat(op, key, Convert.ToSingle(value));
                    break;
                case TF_AttrType.TF_ATTR_SHAPE:
                    var dims = (value as int[]).Select(x => (long)x).ToArray();
                    c_api.TFE_OpSetAttrShape(op, key, dims, dims.Length, status.Handle);
                    status.Check(true);
                    break;
                case TF_AttrType.TF_ATTR_FUNC:
                    if (value is ConcreteFunction func)
                        c_api.TFE_OpSetAttrFunctionName(op, key, func.Name, func.Name.Length);
                    else
                        throw new NotImplementedException("TF_AttrType.TF_ATTR_FUNC");
                    break;
                default:
                    throw new NotImplementedException($"SetOpAttrScalar for {type}");
            }

            return true;
        }
    }
}
