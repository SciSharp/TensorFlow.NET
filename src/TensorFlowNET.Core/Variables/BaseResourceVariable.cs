using Tensorflow.NumPy;
using System;
using Tensorflow.Eager;
using Tensorflow.Variables;
using Tensorflow.Train;
using static Tensorflow.Binding;
using System.Collections.Generic;
using System.Diagnostics;
using Tensorflow.Checkpoint;
using Tensorflow.Training.Saving.SavedModel;
using OneOf;
using Tensorflow.Graphs;

namespace Tensorflow
{
    public class BaseResourceVariable : DisposableTrackableObject
    {
        protected string _name;
        public virtual string Name => _handle_name;
        public virtual string SharedName
        {
            get
            {
                // TODO(Rinne): optimize the implementation with refactor of variable.
                return _handle_name.Substring(0, _handle_name.IndexOf(':') + 1);
            }
        }
        protected TF_DataType _dtype;
        public TF_DataType dtype => _dtype;
        protected string _handle_name;
        public string handle_name
        {
            get { return _handle_name; }
            set { _handle_name = value; }
        }

        protected string _unique_id;
        public string UniqueId => _unique_id;

        protected bool _in_graph_mode;
        internal bool InGraphMode => _in_graph_mode;

        protected bool _trainable;
        public bool Trainable => _trainable;

        protected Tensor _initial_value;

        public Operation initializer => initializer_op;

        protected Tensor _parent_op;
        public Tensor parent_op => _parent_op;

        /// <summary>
        /// Tensor handle
        /// </summary>
        protected Tensor handle;
        public Tensor Handle => handle;
        protected Tensor _graph_element;
        public Tensor GraphElement => _graph_element;
        protected Shape _shape;
        public Shape shape => _shape;

        protected Operation initializer_op;
        public Operation Initializer => initializer_op;
        public Operation Op => handle.op;
        public Graph Graph => handle.graph;
        public string Device => handle.Device;
        EagerResourceDeleter eager_resource_deleter;
        public VariableAggregation Aggregation { get; protected set; } = VariableAggregation.None;

        public BaseResourceVariable()
        {
        }

        public void __init__(bool trainable = true,
            Shape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
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
            if(shape is not null)
            {
                _shape = shape;
            }
            if(dtype != TF_DataType.DtInvalid)
            {
                _dtype = dtype;
            }

            // After the handle has been created, set up a way to clean it up when
            // executing eagerly. We'll hold the only reference to the deleter, so that
            // when this object is garbage collected the deleter will be too. This
            // means ResourceVariables can be part of reference cycles without those
            // cycles being uncollectable.
            if (handle is EagerTensor)
            {
                _handle = handle.EagerTensorHandle.DangerousGetHandle();
                // eager_resource_deleter = new EagerResourceDeleter(handle, handle.Device);
            }
            else if(handle is null)
            {
                // TODO: fix this dangerous change.
                _handle = IntPtr.Zero;
            }
            else
            {
                _handle = handle.Handle == null ? IntPtr.Zero : handle.Handle.DangerousGetHandle();
            }

#if TRACK_TENSOR_LIFE
            print($"Created Resource 0x{_handle.ToString("x16")} {_name}");
#endif
        }

        public Tensor assign<T>(T value, bool use_locking = false, string name = null, bool read_value = true)
        {
            if (value.GetType() == typeof(Tensor))
            {
                var assign = gen_state_ops.assign(handle, value, use_locking: use_locking, name: name);
                if (read_value)
                    return assign;
                return assign.op;
            }

            var value_tensor = ops.convert_to_tensor(value, dtype: dtype);
            var assign_op = gen_resource_variable_ops.assign_variable_op(
                handle, value_tensor, name: name);

            if (read_value)
                return gen_resource_variable_ops.read_variable_op(handle, dtype);

            if (assign_op == null)
                return null;

            return assign_op;
        }

        public void StridedSliceAssign(Tensor value, ParsedSliceArgs slice)
        {
            _strided_slice_assign(slice.PackedBegin, slice.PackedEnd, slice.PackedStrides, value);
        }

        void _strided_slice_assign(Tensor begin, Tensor end, Tensor strides, Tensor value, string name = null,
            int begin_mask = 0, int end_mask = 0, int ellipsis_mask = 0, int new_axis_mask = 0, int shrink_axis_mask = 0)
        {
            var op = gen_array_ops.resource_strided_slice_assign(handle, begin, end, strides, value,
                begin_mask: begin_mask,
                end_mask: end_mask,
                ellipsis_mask: ellipsis_mask,
                new_axis_mask: new_axis_mask,
                shrink_axis_mask: shrink_axis_mask);
        }

        public IVariableV1 assign_lazy_load(Tensor value, string name = null)
        {
            var value_tensor = ops.convert_to_tensor(value, dtype: dtype);
            var assign_op = gen_resource_variable_ops.assign_variable_op(
                handle, value_tensor, name: name);
            var variable = _lazy_read(assign_op, value_tensor);
            return variable;
        }

        public Tensor value()
            => GraphElement ?? _read_variable_op();

        protected Tensor _read_variable_op(bool no_copy = false)
        {
            variable_accessed(this);

            Tensor read_and_set_handle(bool no_copy)
            {
                if (no_copy)
                {
                    gen_resource_variable_ops.disable_copy_on_read(handle);
                }
                var result = gen_resource_variable_ops.read_variable_op(handle, _dtype);
                resource_variable_ops._maybe_set_handle_data(_dtype, handle, result);
                return result;
            }

            // TODO(Rinne): deal with caching device.
            var result = read_and_set_handle(no_copy);
            if (!tf.Context.executing_eagerly())
            {
                tf.Runner.TFE_TapeSetRecordOperation("ReadVariableOp", new Tensor[] { result }, new Tensor[] { handle },
                        backward_function: (x, _) => x);
            }

            // have to set shape when converting to substituent placeholder
            if (result.shape.ndim == -1)
            {
                c_api.TF_GraphSetTensorShape(result.graph,
                    result._as_tf_output(),
                    shape.dims,
                    shape.ndim,
                    tf.Status);
                tf.Status.Check(true);
            }

            return result;
        }

        IVariableV1 _lazy_read(Operation op, Tensor value)
        {
            variable_accessed(this);
            return new _UnreadVariable(handle, _dtype, _shape, _in_graph_mode, _unique_id);
        }

        /// <summary>
        /// Records that `variable` was accessed for the tape and FuncGraph.
        /// </summary>
        void variable_accessed(BaseResourceVariable variable)
        {
            if(ops.get_default_graph() is FuncGraph func_graph)
            {
                func_graph.watch_variable(variable as IVariableV1);
            }
            if (variable.Trainable)
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
        protected Tensor read_value()
        {
            var value = tf_with(ops.name_scope("Read"), delegate
            { 
                return _read_variable_op(); 
            });
            return array_ops.identity(value);
        }
            

        public Tensor assign_add<T>(T delta, bool use_locking = false, string name = null, bool read_value = true)
        {
            var assign_add_op = gen_resource_variable_ops.assign_add_variable_op(Handle,
                ops.convert_to_tensor(delta, dtype: dtype), name: name);

            if (read_value)
                return gen_resource_variable_ops.read_variable_op(handle, dtype);
            // return _lazy_read(assign_add_op);
            return assign_add_op;
        }

        public Tensor assign_sub<T>(T delta, bool use_locking = false, string name = null, bool read_value = true)
        {
            var assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(Handle,
                ops.convert_to_tensor(delta, dtype: dtype), name: name);

            if (read_value)
                return gen_resource_variable_ops.read_variable_op(handle, dtype);
            // return _lazy_read(assign_add_op);
            return assign_sub_op;
        }

        public IVariableV1 assign_sub_lazy_load(Tensor delta, string name = null)
        {
            var assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(Handle,
                ops.convert_to_tensor(delta, dtype: dtype), name: name);

            return _lazy_read(assign_sub_op, delta);
        }

        public override string ToString()
        {
            if (tf.Context.executing_eagerly())
                return $"tf.Variable: '{Name}' shape={string.Join(",", shape)}, dtype={dtype.as_numpy_name()}, numpy={read_value().numpy()}";
            else
                return $"tf.Variable: '{Name}' shape={string.Join(",", shape)}, dtype={dtype.as_numpy_name()}";
        }

        public NDArray numpy() => read_value().numpy();

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
#if TRACK_TENSOR_LIFE
            print($"Deleted Resource 0x{handle.ToString("x16")} {_name}");
#endif
        }

        public Tensor AsTensor(TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
        {
            if (as_ref)
                return read_value().op.inputs[0];
            else
                return value();
        }

        public override (IDictionary<Trackable, Trackable>, IDictionary<Tensor, Tensor>) map_resources(SaveOptions save_options)
        {
            BaseResourceVariable new_variable;
            if (save_options.experimental_variable_policy.save_variable_devices())
            {
                Debug.Assert(this is ResourceVariable);
                new_variable = tf_with(ops.device(this.Device), _ =>
                {
                    return resource_variable_ops.copy_to_graph_uninitialized((ResourceVariable)this);
                });
            }
            else
            {
                new_variable = resource_variable_ops.copy_to_graph_uninitialized((ResourceVariable)this);
            }
            Dictionary<Trackable, Trackable> obj_map = new();
            Dictionary<Tensor, Tensor> resource_map = new();
            obj_map[this] = new_variable;
            resource_map[this.handle] = new_variable.handle;
            return (obj_map, resource_map);
        }

        /// <summary>
        /// Writes additional information of the variable into the SavedObject proto.
        /// ubclasses of ResourceVariables could choose to override this method to
        /// customize extra information to provide when saving a SavedModel.
        /// </summary>
        /// <param name="proto"></param>
        /// <param name="options"></param>
        public virtual void write_object_proto(SavedObject proto, SaveOptions options)
        {
            resource_variable_ops.write_object_proto_for_resource_variable(this, proto, options);
        }

        public override IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> gather_saveables_for_checkpoint()
        {
            var res = new Dictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>>();
            res[Trackable.Constants.VARIABLE_VALUE_KEY] = x => this;
            return res;
        }

        public Tensor is_initialized(string name = null)
        {
            return gen_resource_variable_ops.var_is_initialized_op(this.handle, name);
        }

        public Tensor read_value_no_copy()
        {
            Tensor value = null;
            tf_with(ops.name_scope("Read"), _ =>
            {
                // TODO: `no_copy = true`.
                value = _read_variable_op();
            });
            return array_ops.identity(value);
        }

        //public static Tensor operator +(BaseResourceVariable x, int y) => x.value() + y;
        //public static Tensor operator +(BaseResourceVariable x, float y) => x.value() + y;
        //public static Tensor operator +(BaseResourceVariable x, double y) => x.value() + y;
        //public static Tensor operator +(BaseResourceVariable x, BaseResourceVariable y) => x.value() + y.value();
        //public static Tensor operator -(BaseResourceVariable x, int y) => x.value() - y;
        //public static Tensor operator -(BaseResourceVariable x, float y) => x.value() - y;
        //public static Tensor operator -(BaseResourceVariable x, double y) => x.value() - y;
        //public static Tensor operator -(BaseResourceVariable x, Tensor y) => x.value() - y;
        //public static Tensor operator -(BaseResourceVariable x, BaseResourceVariable y) => x.value() - y.value();

        //public static Tensor operator *(BaseResourceVariable x, BaseResourceVariable y) => x.value() * y.value();
        //public static Tensor operator *(BaseResourceVariable x, Tensor y) => x.value() * y;
        //public static Tensor operator *(BaseResourceVariable x, NDArray y) => x.value() * y;

        //public static Tensor operator <(BaseResourceVariable x, Tensor y) => x.value() < y;

        //public static Tensor operator >(BaseResourceVariable x, Tensor y) => x.value() > y;
    }
}
