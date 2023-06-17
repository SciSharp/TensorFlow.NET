/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow;

public static class gen_resource_variable_ops
{
    /// <summary>
    /// Adds a value to the current value of a variable.
    /// </summary>
    /// <remarks>
    /// 
    /// Any ReadVariableOp with a control dependency on this op is guaranteed to
    /// see the incremented value or a subsequent newer one.
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static Operation assign_add_variable_op(Tensor resource, Tensor value, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AssignAddVariableOp", name) { args = new object[] { resource, value }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return assign_add_variable_op_eager_fallback(resource, value, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["value"] = value;
        var _op = tf.OpDefLib._apply_op_helper("AssignAddVariableOp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype") };
            _execute.record_gradient("AssignAddVariableOp", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation assign_add_variable_op_eager_fallback(Tensor resource, Tensor value, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, value };
        object[] _attrs = new object[] { "dtype", value.dtype };
        var _result = _execute.execute("AssignAddVariableOp", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AssignAddVariableOp", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Subtracts a value from the current value of a variable.
    /// </summary>
    /// <remarks>
    /// 
    /// Any ReadVariableOp with a control dependency on this op is guaranteed to
    /// see the decremented value or a subsequent newer one.
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static Operation assign_sub_variable_op(Tensor resource, Tensor value, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AssignSubVariableOp", name) { args = new object[] { resource, value }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return assign_sub_variable_op_eager_fallback(resource, value, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["value"] = value;
        var _op = tf.OpDefLib._apply_op_helper("AssignSubVariableOp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype") };
            _execute.record_gradient("AssignSubVariableOp", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation assign_sub_variable_op_eager_fallback(Tensor resource, Tensor value, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, value };
        object[] _attrs = new object[] { "dtype", value.dtype };
        var _result = _execute.execute("AssignSubVariableOp", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AssignSubVariableOp", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Assigns a new value to a variable.
    /// </summary>
    /// <remarks>
    /// 
    /// Any ReadVariableOp with a control dependency on this op is guaranteed to return
    /// this value or a subsequent newer value of the variable.
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="value"></param>
    /// <param name="validate_shape"></param>
    /// <returns></returns>
    public static Operation assign_variable_op(Tensor resource, Tensor value, bool validate_shape = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AssignVariableOp", name) { args = new object[] { resource, value }, attrs = new Dictionary<string, object>() { ["validate_shape"] = validate_shape } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return assign_variable_op_eager_fallback(resource, value, validate_shape: validate_shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["value"] = value;
        keywords["validate_shape"] = validate_shape;
        var _op = tf.OpDefLib._apply_op_helper("AssignVariableOp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "validate_shape", _op._get_attr_bool("validate_shape") };
            _execute.record_gradient("AssignVariableOp", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation assign_variable_op_eager_fallback(Tensor resource, Tensor value, bool validate_shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, value };
        object[] _attrs = new object[] { "dtype", value.dtype, "validate_shape", validate_shape };
        var _result = _execute.execute("AssignVariableOp", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AssignVariableOp", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// This op consumes a lock created by `MutexLock`.
    /// </summary>
    /// <remarks>
    /// 
    /// This op exists to consume a tensor created by `MutexLock` (other than
    /// direct control dependencies).  It should be the only that consumes the tensor,
    /// and will raise an error if it is not.  Its only purpose is to keep the
    /// mutex lock tensor alive until it is consumed by this op.
    /// 
    /// **NOTE**: This operation must run on the same device as its input.  This may
    /// be enforced via the `colocate_with` mechanism.
    /// 
    /// </remarks>
    /// <param name="mutex_lock"></param>
    /// <returns></returns>
    public static Operation consume_mutex_lock(Tensor mutex_lock, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ConsumeMutexLock", name) { args = new object[] { mutex_lock }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return consume_mutex_lock_eager_fallback(mutex_lock, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["mutex_lock"] = mutex_lock;
        var _op = tf.OpDefLib._apply_op_helper("ConsumeMutexLock", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ConsumeMutexLock", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation consume_mutex_lock_eager_fallback(Tensor mutex_lock, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { mutex_lock };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ConsumeMutexLock", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ConsumeMutexLock", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Deletes the resource specified by the handle.
    /// </summary>
    /// <remarks>
    /// 
    /// All subsequent operations using the resource will result in a NotFound
    /// error status.
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="ignore_lookup_error">
    /// 
    /// whether to ignore the error when the resource
    /// doesn't exist.
    /// 
    /// </param>
    /// <returns></returns>
    public static Operation destroy_resource_op(Tensor resource, bool ignore_lookup_error = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DestroyResourceOp", name) { args = new object[] { resource }, attrs = new Dictionary<string, object>() { ["ignore_lookup_error"] = ignore_lookup_error } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return destroy_resource_op_eager_fallback(resource, ignore_lookup_error: ignore_lookup_error, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["ignore_lookup_error"] = ignore_lookup_error;
        var _op = tf.OpDefLib._apply_op_helper("DestroyResourceOp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ignore_lookup_error", _op._get_attr_bool("ignore_lookup_error") };
            _execute.record_gradient("DestroyResourceOp", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation destroy_resource_op_eager_fallback(Tensor resource, bool ignore_lookup_error, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource };
        object[] _attrs = new object[] { "ignore_lookup_error", ignore_lookup_error };
        var _result = _execute.execute("DestroyResourceOp", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DestroyResourceOp", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Turns off the copy-on-read mode.
    /// </summary>
    /// <remarks>
    /// 
    /// Turns off the copy-on-read mode of a resource variable. If the variable is not in copy-on-read mode, this op has no effect.  
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <returns></returns>
    public static Operation disable_copy_on_read(Tensor resource, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DisableCopyOnRead", name) { args = new object[] { resource }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return disable_copy_on_read_eager_fallback(resource, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        var _op = tf.OpDefLib._apply_op_helper("DisableCopyOnRead", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("DisableCopyOnRead", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation disable_copy_on_read_eager_fallback(Tensor resource, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("DisableCopyOnRead", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DisableCopyOnRead", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Locks a mutex resource.  The output is the lock.  So long as the lock tensor
    /// </summary>
    /// <remarks>
    /// 
    /// is alive, any other request to use `MutexLock` with this mutex will wait.
    /// 
    /// This is particularly useful for creating a critical section when used in
    /// conjunction with `MutexLockIdentity`:
    /// 
    /// ```python
    /// 
    /// mutex = mutex_v2(
    ///   shared_name=handle_name, container=container, name=name)
    /// 
    /// def execute_in_critical_section(fn, *args, **kwargs):
    ///   lock = gen_resource_variable_ops.mutex_lock(mutex)
    /// 
    ///   with ops.control_dependencies([lock]):
    ///     r = fn(*args, **kwargs)
    /// 
    ///   with ops.control_dependencies(nest.flatten(r)):
    ///     with ops.colocate_with(mutex):
    ///       ensure_lock_exists = mutex_lock_identity(lock)
    /// 
    ///     # Make sure that if any element of r is accessed, all of
    ///     # them are executed together.
    ///     r = nest.map_structure(tf.identity, r)
    /// 
    ///   with ops.control_dependencies([ensure_lock_exists]):
    ///     return nest.map_structure(tf.identity, r)
    /// ```
    /// 
    /// While `fn` is running in the critical section, no other functions which wish to
    /// use this critical section may run.
    /// 
    /// Often the use case is that two executions of the same graph, in parallel,
    /// wish to run `fn`; and we wish to ensure that only one of them executes
    /// at a time.  This is especially important if `fn` modifies one or more
    /// variables at a time.
    /// 
    /// It is also useful if two separate functions must share a resource, but we
    /// wish to ensure the usage is exclusive.
    /// 
    /// </remarks>
    /// <param name="mutex"></param>
    /// <returns></returns>
    public static Tensor mutex_lock(Tensor mutex, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MutexLock", name) { args = new object[] { mutex }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return mutex_lock_eager_fallback(mutex, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["mutex"] = mutex;
        var _op = tf.OpDefLib._apply_op_helper("MutexLock", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("MutexLock", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mutex_lock_eager_fallback(Tensor mutex, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { mutex };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("MutexLock", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MutexLock", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Creates a Mutex resource that can be locked by `MutexLock`.
    /// </summary>
    /// <param name="container">
    /// 
    /// If non-empty, this variable is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this variable is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor mutex_v2(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MutexV2", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return mutex_v2_eager_fallback(container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("MutexV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("MutexV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mutex_v2_eager_fallback(string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "container", container, "shared_name", shared_name };
        var _result = _execute.execute("MutexV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MutexV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Reads the value of a variable.
    /// </summary>
    /// <remarks>
    /// 
    /// The tensor returned by this operation is immutable.
    /// 
    /// The value returned by this operation is guaranteed to be influenced by all the
    /// writes on which this operation depends directly or indirectly, and to not be
    /// influenced by any of the writes which depend directly or indirectly on this
    /// operation.
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="dtype">
    /// 
    /// the dtype of the value.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor read_variable_op(Tensor resource, TF_DataType dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReadVariableOp", name) { args = new object[] { resource }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return read_variable_op_eager_fallback(resource, dtype: dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["dtype"] = dtype;
        var _op = tf.OpDefLib._apply_op_helper("ReadVariableOp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype") };
            _execute.record_gradient("ReadVariableOp", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor read_variable_op_eager_fallback(Tensor resource, TF_DataType dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource };
        object[] _attrs = new object[] { "dtype", dtype };
        var _result = _execute.execute("ReadVariableOp", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReadVariableOp", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Gather slices from the variable pointed to by `resource` according to `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
    /// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
    /// 
    /// ```python
    ///     # Scalar indices
    ///     output[:, ..., :] = params[indices, :, ... :]
    /// 
    ///     # Vector indices
    ///     output[i, :, ..., :] = params[indices[i], :, ... :]
    /// 
    ///     # Higher rank indices
    ///     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
    /// ```
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="batch_dims"></param>
    /// <param name="validate_indices"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    public static Tensor resource_gather(Tensor resource, Tensor indices, TF_DataType dtype, int batch_dims = 0, bool validate_indices = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceGather", name) { args = new object[] { resource, indices }, attrs = new Dictionary<string, object>() { ["batch_dims"] = batch_dims, ["validate_indices"] = validate_indices, ["dtype"] = dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_gather_eager_fallback(resource, indices, batch_dims: batch_dims, validate_indices: validate_indices, dtype: dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["batch_dims"] = batch_dims;
        keywords["validate_indices"] = validate_indices;
        keywords["dtype"] = dtype;
        var _op = tf.OpDefLib._apply_op_helper("ResourceGather", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "batch_dims", _op._get_attr_int("batch_dims"), "validate_indices", _op._get_attr_bool("validate_indices"), "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceGather", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor resource_gather_eager_fallback(Tensor resource, Tensor indices, int batch_dims, bool validate_indices, TF_DataType dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices };
        object[] _attrs = new object[] { "batch_dims", batch_dims, "validate_indices", validate_indices, "dtype", dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceGather", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceGather", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    public static Tensor resource_gather_nd(Tensor resource, Tensor indices, TF_DataType dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceGatherNd", name) { args = new object[] { resource, indices }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_gather_nd_eager_fallback(resource, indices, dtype: dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["dtype"] = dtype;
        var _op = tf.OpDefLib._apply_op_helper("ResourceGatherNd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceGatherNd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor resource_gather_nd_eager_fallback(Tensor resource, Tensor indices, TF_DataType dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices };
        object[] _attrs = new object[] { "dtype", dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceGatherNd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceGatherNd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Adds sparse updates to the variable referenced by `resource`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes
    /// 
    ///     # Scalar indices
    ///     ref[indices, ...] += updates[...]
    /// 
    ///     # Vector indices (for each i)
    ///     ref[indices[i], ...] += updates[i, ...]
    /// 
    ///     # High rank indices (for each i, ..., j)
    ///     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
    /// 
    /// Duplicate entries are handled correctly: if multiple `indices` reference
    /// the same location, their contributions add.
    /// 
    /// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
    /// </div>
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Operation resource_scatter_add(Tensor resource, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceScatterAdd", name) { args = new object[] { resource, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_scatter_add_eager_fallback(resource, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ResourceScatterAdd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceScatterAdd", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_scatter_add_eager_fallback(Tensor resource, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices, updates };
        object[] _attrs = new object[] { "dtype", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceScatterAdd", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceScatterAdd", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Divides sparse updates into the variable referenced by `resource`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes
    /// 
    ///     # Scalar indices
    ///     ref[indices, ...] /= updates[...]
    /// 
    ///     # Vector indices (for each i)
    ///     ref[indices[i], ...] /= updates[i, ...]
    /// 
    ///     # High rank indices (for each i, ..., j)
    ///     ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]
    /// 
    /// Duplicate entries are handled correctly: if multiple `indices` reference
    /// the same location, their contributions multiply.
    /// 
    /// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
    /// </div>
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Operation resource_scatter_div(Tensor resource, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceScatterDiv", name) { args = new object[] { resource, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_scatter_div_eager_fallback(resource, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ResourceScatterDiv", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceScatterDiv", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_scatter_div_eager_fallback(Tensor resource, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices, updates };
        object[] _attrs = new object[] { "dtype", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceScatterDiv", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceScatterDiv", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Reduces sparse updates into the variable referenced by `resource` using the `max` operation.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes
    /// 
    ///     # Scalar indices
    ///     ref[indices, ...] = max(ref[indices, ...], updates[...])
    /// 
    ///     # Vector indices (for each i)
    ///     ref[indices[i], ...] = max(ref[indices[i], ...], updates[i, ...])
    /// 
    ///     # High rank indices (for each i, ..., j)
    ///     ref[indices[i, ..., j], ...] = max(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
    /// 
    /// Duplicate entries are handled correctly: if multiple `indices` reference
    /// the same location, their contributions are combined.
    /// 
    /// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
    /// </div>
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Operation resource_scatter_max(Tensor resource, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceScatterMax", name) { args = new object[] { resource, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_scatter_max_eager_fallback(resource, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ResourceScatterMax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceScatterMax", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_scatter_max_eager_fallback(Tensor resource, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices, updates };
        object[] _attrs = new object[] { "dtype", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceScatterMax", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceScatterMax", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Reduces sparse updates into the variable referenced by `resource` using the `min` operation.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes
    /// 
    ///     # Scalar indices
    ///     ref[indices, ...] = min(ref[indices, ...], updates[...])
    /// 
    ///     # Vector indices (for each i)
    ///     ref[indices[i], ...] = min(ref[indices[i], ...], updates[i, ...])
    /// 
    ///     # High rank indices (for each i, ..., j)
    ///     ref[indices[i, ..., j], ...] = min(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
    /// 
    /// Duplicate entries are handled correctly: if multiple `indices` reference
    /// the same location, their contributions are combined.
    /// 
    /// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
    /// </div>
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Operation resource_scatter_min(Tensor resource, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceScatterMin", name) { args = new object[] { resource, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_scatter_min_eager_fallback(resource, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ResourceScatterMin", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceScatterMin", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_scatter_min_eager_fallback(Tensor resource, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices, updates };
        object[] _attrs = new object[] { "dtype", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceScatterMin", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceScatterMin", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Multiplies sparse updates into the variable referenced by `resource`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes
    /// 
    ///     # Scalar indices
    ///     ref[indices, ...] *= updates[...]
    /// 
    ///     # Vector indices (for each i)
    ///     ref[indices[i], ...] *= updates[i, ...]
    /// 
    ///     # High rank indices (for each i, ..., j)
    ///     ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]
    /// 
    /// Duplicate entries are handled correctly: if multiple `indices` reference
    /// the same location, their contributions multiply.
    /// 
    /// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
    /// </div>
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Operation resource_scatter_mul(Tensor resource, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceScatterMul", name) { args = new object[] { resource, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_scatter_mul_eager_fallback(resource, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ResourceScatterMul", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceScatterMul", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_scatter_mul_eager_fallback(Tensor resource, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices, updates };
        object[] _attrs = new object[] { "dtype", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceScatterMul", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceScatterMul", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Subtracts sparse updates from the variable referenced by `resource`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes
    /// 
    ///     # Scalar indices
    ///     ref[indices, ...] -= updates[...]
    /// 
    ///     # Vector indices (for each i)
    ///     ref[indices[i], ...] -= updates[i, ...]
    /// 
    ///     # High rank indices (for each i, ..., j)
    ///     ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
    /// 
    /// Duplicate entries are handled correctly: if multiple `indices` reference
    /// the same location, their contributions add.
    /// 
    /// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
    /// </div>
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Operation resource_scatter_sub(Tensor resource, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceScatterSub", name) { args = new object[] { resource, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_scatter_sub_eager_fallback(resource, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ResourceScatterSub", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceScatterSub", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_scatter_sub_eager_fallback(Tensor resource, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices, updates };
        object[] _attrs = new object[] { "dtype", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceScatterSub", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceScatterSub", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Assigns sparse updates to the variable referenced by `resource`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes
    /// 
    ///     # Scalar indices
    ///     ref[indices, ...] = updates[...]
    /// 
    ///     # Vector indices (for each i)
    ///     ref[indices[i], ...] = updates[i, ...]
    /// 
    ///     # High rank indices (for each i, ..., j)
    ///     ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
    /// 
    /// </remarks>
    /// <param name="resource"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Operation resource_scatter_update(Tensor resource, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceScatterUpdate", name) { args = new object[] { resource, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_scatter_update_eager_fallback(resource, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ResourceScatterUpdate", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ResourceScatterUpdate", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_scatter_update_eager_fallback(Tensor resource, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource, indices, updates };
        object[] _attrs = new object[] { "dtype", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ResourceScatterUpdate", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceScatterUpdate", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Creates a handle to a Variable resource.
    /// </summary>
    /// <param name="container">
    /// 
    /// the container this variable is placed in.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// the name by which this variable is referred to.
    /// 
    /// </param>
    /// <param name="dtype">
    /// 
    /// the type of this variable. Must agree with the dtypes
    /// of all ops using this variable.
    /// 
    /// </param>
    /// <param name="shape">
    /// 
    /// The (possibly partially specified) shape of this variable.
    /// 
    /// </param>
    /// <param name="allowed_devices">
    /// 
    /// DEPRECATED. The allowed devices containing the resource variable. Set when the
    /// output ResourceHandle represents a per-replica/partitioned resource variable.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor var_handle_op(TF_DataType dtype, Shape shape, string container = "", string shared_name = "", string[] allowed_devices = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (allowed_devices is null)
        {
            allowed_devices = new string[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "VarHandleOp", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["container"] = container, ["shared_name"] = shared_name, ["dtype"] = dtype, ["shape"] = shape, ["allowed_devices"] = allowed_devices } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return var_handle_op_eager_fallback(container: container, shared_name: shared_name, dtype: dtype, shape: shape, allowed_devices: allowed_devices, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        keywords["dtype"] = dtype;
        keywords["shape"] = shape;
        keywords["allowed_devices"] = allowed_devices;
        var _op = tf.OpDefLib._apply_op_helper("VarHandleOp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name"), "dtype", _op._get_attr_type("dtype"), "shape", _op.get_attr("shape"), "allowed_devices", _op.get_attr("allowed_devices") };
            _execute.record_gradient("VarHandleOp", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor var_handle_op_eager_fallback(string container, string shared_name, TF_DataType dtype, Shape shape, string[] allowed_devices, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "container", container, "shared_name", shared_name, "dtype", dtype, "shape", shape, "allowed_devices", allowed_devices };
        var _result = _execute.execute("VarHandleOp", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("VarHandleOp", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Checks whether a resource handle-based variable has been initialized.
    /// </summary>
    /// <param name="resource"></param>
    /// <returns></returns>
    public static Tensor var_is_initialized_op(Tensor resource, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "VarIsInitializedOp", name) { args = new object[] { resource }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return var_is_initialized_op_eager_fallback(resource, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["resource"] = resource;
        var _op = tf.OpDefLib._apply_op_helper("VarIsInitializedOp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("VarIsInitializedOp", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor var_is_initialized_op_eager_fallback(Tensor resource, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { resource };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("VarIsInitializedOp", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("VarIsInitializedOp", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the shape of the variable pointed to by `resource`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns a 1-D integer tensor representing the shape of `input`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    /// shape(t) ==> [2, 2, 3]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor variable_shape(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "VariableShape", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return variable_shape_eager_fallback(input, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("VariableShape", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("VariableShape", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor variable_shape_eager_fallback(Tensor input, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "out_type", out_type };
        var _result = _execute.execute("VariableShape", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("VariableShape", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
}
