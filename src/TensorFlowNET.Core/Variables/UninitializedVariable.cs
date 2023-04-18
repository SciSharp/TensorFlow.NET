using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Gradients;
using static Tensorflow.Binding;

namespace Tensorflow.Variables
{
    /// <summary>
    /// A variable with no initializer.
    /// </summary>
    public sealed class UninitializedVariable : BaseResourceVariable, IVariableV1
    {
        // TODO: complete the arg list.
        public UninitializedVariable(
            bool trainable = true,
            string caching_device = "",
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            VariableAggregation aggregation = VariableAggregation.None,
            Shape shape = null,
            Tensor extra_handle_data = null)
        {
            string unique_id = "";
            string handle_name = "";
            Tensor created_handle = null;
            tf_with(ops.init_scope(), (x) =>
            {
                _in_graph_mode = !tf.Context.executing_eagerly();
                tf_with(ops.name_scope(name, "Variable", skip_on_eager: false), name =>
                {
                    handle_name = ops.name_from_scope_name(name);
                    string? shared_name;
                    if (_in_graph_mode)
                    {
                        shared_name = handle_name;
                        unique_id = shared_name;
                    }
                    else
                    {
                        unique_id = $"{handle_name}-{ops.uid()}";
                        shared_name = null;
                    }
                    created_handle = resource_variable_ops.variable_handle_from_shape_and_dtype(
                        shape, dtype, shared_name, name, _in_graph_mode, extra_handle_data);
                    // skip the assignment of `handle._parent_trackable` because of lack of API.
                    // skip the assignment of `handle._name` and `handle._unique_id` because of accessability.

                    if (_in_graph_mode)
                    {
                        tf_with(ops.name_scope("Read"), _ =>
                        {
                            var value = tf_with(ops.device(created_handle.Device), _ =>
                            {
                                var result = gen_resource_variable_ops.read_variable_op(created_handle, dtype);
                                resource_variable_ops._maybe_set_handle_data(dtype, created_handle, result);
                                return result;
                            });
                            _graph_element = value;
                        });
                        ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES_, this);
                    }
                    else
                    {
                        _graph_element = null;
                    }
                });
            });
            base.__init__(trainable, shape, dtype, created_handle, unique_id: unique_id, handle_name: handle_name);
        }
    }
}
