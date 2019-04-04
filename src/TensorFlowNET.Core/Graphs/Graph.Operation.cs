using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Graph
    {
        public OpDef GetOpDef(string type)
        {
            using (var buffer = new Buffer())
            using (var status = new Status())
            {
                c_api.TF_GraphGetOpDef(_handle, type, buffer, status);
                return OpDef.Parser.ParseFrom(buffer.Data);
            }
        }

        public OperationDescription NewOperation(string opType, string opName)
        {
            return c_api.TF_NewOperation(_handle, opType, opName);
        }

        public Operation get_operation_by_name(string name) 
            => as_graph_element(name, allow_tensor: false, allow_operation: true) as Operation;

        public ITensorOrOperation _get_operation_by_name_unsafe(string name)
        {
            return _nodes_by_name.ContainsKey(name) ? _nodes_by_name[name] : null;
        }

        public ITensorOrOperation _get_operation_by_tf_operation(IntPtr tf_oper)
        {
            var op_name = Marshal.PtrToStringAnsi(c_api.TF_OperationName(tf_oper));
            return _get_operation_by_name_unsafe(op_name);
        }

        public Operation _create_op_from_tf_operation(IntPtr c_op, bool compute_device = true)
        {
            var ret = new Operation(c_op);
            _add_op(ret);

            var name_key = ret.name.ToLower();
            if (!_names_in_use.ContainsKey(name_key))
                _names_in_use[name_key] = 1;

            _create_op_helper(ret, compute_device: compute_device);

            return ret;
        }

        /// <summary>
        /// Creates `Operations` in this graph for any new TF_Operations.
        /// 
        /// This is useful for when TF_Operations are indirectly created by the C API
        /// outside of the Operation constructor (e.g. by TF_ImportGraphDef,
        /// TF_FinishWhile). This ensures there are corresponding Operations for all
        /// TF_Operations in the underlying TF_Graph.
        /// </summary>
        /// <param name="compute_devices"></param>
        /// <returns></returns>
        public IEnumerable<Operation> _add_new_tf_operations(bool compute_devices = true)
        {
            var new_ops = c_api_util.new_tf_operations(this)
                .Select(c_op => _create_op_from_tf_operation(c_op, compute_device: compute_devices))
                .ToArray();

            foreach(var op in new_ops)
            {
                var new_control_inputs = _control_dependencies_for_inputs(op.inputs)
                    .Select(x => x as Operation)
                    .ToArray();
                op._add_control_inputs(new_control_inputs);
                op._control_flow_post_processing();
            }

            return new_ops;
        }
    }
}
