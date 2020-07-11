using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using Tensorflow.Gradients;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class BaseResourceVariable : DisposableObject, IVariableV1
    {
        protected string _name;
        public virtual string Name => _handle_name;
        protected TF_DataType _dtype;
        public TF_DataType dtype => _dtype;
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

        /// <summary>
        /// Tensor handle
        /// </summary>
        protected Tensor handle;
        public Tensor Handle => handle;
        protected Tensor _graph_element;
        public Tensor GraphElement => _graph_element;
        protected TensorShape _shape;
        public TensorShape shape => _shape;

        protected Operation initializer_op;
        public Operation Initializer => initializer_op;
        public Operation Op => handle.op;
        public Graph Graph => handle.graph;

        public BaseResourceVariable()
        {
        }

        public BaseResourceVariable(IntPtr handle, IntPtr tensor)
        {
            _handle = handle;
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
            this.handle = handle;
            _name = name;

            // handle_deleter
        }

        public ITensorOrOperation assign<T>(T value, bool use_locking = false, string name = null, bool read_value = true)
        {
            var value_tensor = ops.convert_to_tensor(value, dtype: dtype);
            var assign_op = gen_resource_variable_ops.assign_variable_op(
                handle, value_tensor, name: name);
            if (read_value)
                return gen_resource_variable_ops.read_variable_op(handle, dtype);
                // return _lazy_read(assign_op, value_tensor);
            return assign_op;
        }

        public Tensor value() => _read_variable_op();

        protected Tensor _read_variable_op()
        {
            variable_accessed(this);
            var result = gen_resource_variable_ops.read_variable_op(handle, _dtype);
            // _maybe_set_handle_data(_dtype, _handle, result);
            return result;
        }

        BaseResourceVariable _lazy_read(Operation op, Tensor value)
        {
            variable_accessed(this);
            return new _UnreadVariable(handle, _dtype, _shape, _in_graph_mode, _unique_id);
        }

        /// <summary>
        /// Records that `variable` was accessed for the tape and FuncGraph.
        /// </summary>
        void variable_accessed(BaseResourceVariable variable)
        {
            if (variable.trainable)
            {
                foreach (var tape in tf.GetTapeSet())
                    tape.VariableAccessed(variable as ResourceVariable);
            }
        }

        /// <summary>
        /// Constructs an op which reads the value of this variable.
        /// 
        /// Should be used when there are multiple reads, or when it is desirable to
        /// read the value only after some condition is true.
        /// </summary>
        /// <returns></returns>
        Tensor read_value()
            => tf_with(ops.name_scope("Read"), delegate
            {
                var value = _read_variable_op();
                return array_ops.identity(value);
            });

        public ITensorOrOperation assign_add<T>(T delta, bool use_locking = false, string name = null, bool read_value = true)
        {
            var assign_add_op = gen_resource_variable_ops.assign_add_variable_op(Handle,
                ops.convert_to_tensor(delta, dtype: dtype), name: name);
            
            if (read_value)
                return gen_resource_variable_ops.read_variable_op(handle, dtype);
                // return _lazy_read(assign_add_op);
            return assign_add_op;
        }

        public override string ToString()
        {
            if (tf.context.executing_eagerly())
                return $"tf.Variable: '{Name}' shape={string.Join(",", shape)}, dtype={dtype.as_numpy_name()}, numpy={EagerTensor.GetFormattedString(dtype, numpy())}";
            else
                return $"tf.Variable: '{Name}' shape={string.Join(",", shape)}, dtype={dtype.as_numpy_name()}";
        }

        public NDArray numpy() => read_value().numpy();

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
        }

        public Tensor AsTensor() => _graph_element;
    }
}
