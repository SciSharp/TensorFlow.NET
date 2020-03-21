using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class BaseResourceVariable : VariableV1
    {
        protected string _handle_name;
        protected string handle_name => _handle_name;

        protected string _unique_id;
        public string unique_id => _unique_id;

        protected bool _in_graph_mode;

        protected bool _trainable;
        public bool trainable => _trainable;

        protected Tensor _initial_value;
        public Tensor initial_value => _initial_value;

        protected Tensor _parent_op;
        public Tensor parent_op => _parent_op;

        protected Tensor _handle;
        /// <summary>
        /// Variable handle
        /// </summary>
        public Tensor handle => _handle;

        protected TensorShape _shape;
        public TensorShape shape => _shape;

        public BaseResourceVariable() : base()
        {
        }

        public void __init__(bool trainable = true,
            Tensor handle = null,
            string name = null,
            string unique_id = null,
            string handle_name = null)
        {
            _trainable = trainable;
            _handle_name = handle_name + ":0";
            _unique_id = unique_id;
            _handle = handle;
            _name = name;
        }

        public override BaseResourceVariable assign(object value, bool use_locking = false, string name = null, bool read_value = true)
        {
            var value_tensor = ops.convert_to_tensor(value, dtype: dtype);
            var assign_op = gen_resource_variable_ops.assign_variable_op(
                _handle, value_tensor, name: name);
            if (read_value)
                return _lazy_read(assign_op, value_tensor);
            return null;
        }

        public Tensor value() => _read_variable_op();

        protected Tensor _read_variable_op()
        {
            var result = gen_resource_variable_ops.read_variable_op(_handle, _dtype);
            // _maybe_set_handle_data(_dtype, _handle, result);
            return result;
        }

        BaseResourceVariable _lazy_read(Operation op, Tensor value)
        {
            variable_accessed(this);
            return new _UnreadVariable(_handle, _dtype, _shape, _in_graph_mode, _unique_id);
        }

        /// <summary>
        /// Records that `variable` was accessed for the tape and FuncGraph.
        /// </summary>
        void variable_accessed(BaseResourceVariable variable)
        {
            if (variable.trainable)
                ; // tape.variable_accessed(variable)
        }

        public override string ToString()
            => $"tf.Variable '{name}' shape={shape} dtype={dtype.as_numpy_name()}, numpy={numpy()}";

        public NDArray numpy() => _read_variable_op().numpy();
    }
}
