using NumSharp;
using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Constraints;
using Tensorflow.Keras.Initializers;
using Tensorflow.Keras.Losses;
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

        public Tensor[] weights
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

        public Layer(bool trainable = true, string name = null, string dtype = null, bool @dynamic = false, Dictionary<string, object> kwargs = null)
        {

        }

        public void build(TensorShape shape) => throw new NotImplementedException();

        public void call(Tensor[] inputs) => throw new NotImplementedException();

        public void _add_trackable(dynamic trackable_object, bool trainable) => throw new NotImplementedException();

        public void add_weight(string name= null, TensorShape shape= null, string dtype= null, Initializer initializer = null,
                               Regularizer regularizer = null, bool? trainable = null, ConstraintBase constraint = null,
                                dynamic partitioner= null, bool? use_resource= null, VariableSynchronization synchronization= VariableSynchronization.Auto,
                                VariableAggregation aggregation= VariableAggregation.None, Dictionary<string, object> kwargs = null) => throw new NotImplementedException();

        public Dictionary<string, object> get_config() => throw new NotImplementedException();

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
    }
}
