using NumSharp;
using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Constraints;
using Tensorflow.Keras.Initializers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Regularizers;

namespace Keras.Layers
{
    public abstract class Layer
    {
        public TF_DataType dtype
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public string name
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public bool stateful
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        public bool trainable
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Regularizer activity_regularizer
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        public dynamic input_spec
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] trainable_weights
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] non_trainable_weights
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private Tensor[] _weights
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Func<bool>[] updates
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] losses
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] metrics
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] input_mask
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] output_mask
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] input
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] output
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public TensorShape[] input_shape
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public TensorShape[] output_shape
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Tensor[] variables
        {
            get
            {
                return _weights;
            }
        }

        public Tensor[] trainable_variables
        {
            get
            {
                return trainable_weights;
            }
        }

        public Tensor[] non_trainable_variables
        {
            get
            {
                return non_trainable_weights;
            }
        }

        private string _compute_dtype
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Layer(bool trainable = true, string name = null, string dtype = null, bool @dynamic = false, Dictionary<string, object> kwargs = null)
        {

        }

        public void build(TensorShape shape) => throw new NotImplementedException();

        public virtual void call(Tensor[] inputs) => throw new NotImplementedException();

        public void _add_trackable(dynamic trackable_object, bool trainable) => throw new NotImplementedException();

        public void add_weight(string name= null, TensorShape shape= null, string dtype= null, Initializer initializer = null,
                               Regularizer regularizer = null, bool? trainable = null, ConstraintBase constraint = null,
                                dynamic partitioner= null, bool? use_resource= null, VariableSynchronization synchronization= VariableSynchronization.Auto,
                                VariableAggregation aggregation= VariableAggregation.None, Dictionary<string, object> kwargs = null) => throw new NotImplementedException();

        public virtual Dictionary<string, object> get_config() => throw new NotImplementedException();

        public Layer from_config(Dictionary<string, object> config) => throw new NotImplementedException();

        public TensorShape compute_output_shape(TensorShape input_shape) => throw new NotImplementedException();

        public dynamic compute_output_signature(dynamic input_signature) => throw new NotImplementedException();

        public Tensor[] compute_mask(Tensor[] inputs, Tensor[] mask = null) => throw new NotImplementedException();

        public void __call__(Tensor[] inputs) => throw new NotImplementedException();

        public void add_loss(Loss[] losses, Tensor[] inputs = null) => throw new NotImplementedException();

        public void _clear_losses() => throw new NotImplementedException();

        public void add_metric(Tensor value, string aggregation= null, string name= null) => throw new NotImplementedException();

        public void add_update(Func<bool>[] updates) => throw new NotImplementedException();

        public void set_weights(NDArray[] weights) => throw new NotImplementedException();

        public NDArray[] get_weights() => throw new NotImplementedException();

        public Func<bool>[] get_updates_for(Tensor[] inputs) => throw new NotImplementedException();

        public Tensor[] get_losses_for(Tensor[] inputs) => throw new NotImplementedException();

        public Tensor[] get_input_mask_at(int node_index) => throw new NotImplementedException();

        public Tensor[] get_output_mask_at(int node_index) => throw new NotImplementedException();

        public TensorShape[] get_input_shape_at(int node_index) => throw new NotImplementedException();

        public TensorShape[] get_output_shape_at(int node_index) => throw new NotImplementedException();

        public Tensor[] get_input_at(int node_index) => throw new NotImplementedException();

        public Tensor[] get_output_at(int node_index) => throw new NotImplementedException();

        public int count_params() => throw new NotImplementedException();

        private void _set_dtype_policy(string dtype) => throw new NotImplementedException();

        private Tensor _maybe_cast_inputs(Tensor inputs) => throw new NotImplementedException();

        private void _warn_about_input_casting(string input_dtype) => throw new NotImplementedException();

        private string _name_scope()
        {
            return name;
        }

        private string _obj_reference_counts
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private dynamic _attribute_sentinel
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private dynamic _call_full_argspec
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private string[] _call_fn_args
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private string[] _call_accepts_kwargs
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private bool _should_compute_mask
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private Tensor[] _eager_losses
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        private dynamic _trackable_saved_model_saver
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private string _object_identifier
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        private string _tracking_metadata
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Dictionary<string, bool> state
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }

        private void _init_set_name(string name, bool zero_based= true) => throw new NotImplementedException();

        private Metric _get_existing_metric(string name = null) => throw new NotImplementedException();

        private void _eager_add_metric(Metric value, string aggregation= null, string name= null) => throw new NotImplementedException();

        private void _symbolic_add_metric(Metric value, string aggregation = null, string name = null) => throw new NotImplementedException();

        private void _handle_weight_regularization(string name, VariableV1 variable, Regularizer regularizer) => throw new NotImplementedException();

        private void _handle_activity_regularization(Tensor[] inputs, Tensor[] outputs) => throw new NotImplementedException();

        private void _set_mask_metadata(Tensor[] inputs, Tensor[]  outputs, Tensor previous_mask) => throw new NotImplementedException();

        private Tensor[] _collect_input_masks(Tensor[] inputs, Dictionary<string, object> args, Dictionary<string, object> kwargs) => throw new NotImplementedException();

        private bool _call_arg_was_passed(string arg_name, Dictionary<string, object> args, Dictionary<string, object> kwargs, bool inputs_in_args= false) => throw new NotImplementedException();

        private T _get_call_arg_value<T>(string arg_name, Dictionary<string, object> args, Dictionary<string, object> kwargs, bool inputs_in_args = false) => throw new NotImplementedException();

        private (Tensor[], Tensor[]) _set_connectivity_metadata_(Tensor[] inputs, Tensor[] outputs, Dictionary<string, object> args, Dictionary<string, object> kwargs) => throw new NotImplementedException();

        private void _add_inbound_node(Tensor[] input_tensors, Tensor[] output_tensors, Dictionary<string, object> args = null) => throw new NotImplementedException();

        private AttrValue _get_node_attribute_at_index(int node_index, string attr, string attr_name) => throw new NotImplementedException();

        private void _maybe_build(Tensor[] inputs) => throw new NotImplementedException();

        private void _symbolic_call(Tensor[] inputs) => throw new NotImplementedException();

        private Dictionary<Layer, bool> _get_trainable_state() => throw new NotImplementedException();

        private void _set_trainable_state(bool trainable_state) => throw new NotImplementedException();

        private void _maybe_create_attribute(string name, object default_value) => throw new NotImplementedException();

        private void __delattr__(string name) => throw new NotImplementedException();

        private void __setattr__(string name, object value) => throw new NotImplementedException();

        private List<AttrValue> _gather_children_attribute(string attribute) => throw new NotImplementedException();

        private List<Layer> _gather_unique_layers() => throw new NotImplementedException();

        private List<Layer> _gather_layers() => throw new NotImplementedException();

        private bool _is_layer() => throw new NotImplementedException();

        private void _init_call_fn_args() => throw new NotImplementedException();

        public dynamic _list_extra_dependencies_for_serialization(dynamic serialization_cache) => throw new NotImplementedException();

        public dynamic _list_functions_for_serialization(dynamic serialization_cache) => throw new NotImplementedException();
    } 
}
