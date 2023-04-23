using System;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        protected virtual IVariableV1 add_weight(string name,
            Shape shape,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            IInitializer initializer = null,
            IRegularizer regularizer = null,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None,
            bool trainable = true,
            Func<VariableArgs, IVariableV1> getter = null)
        {
            // Initialize variable when no initializer provided
            if (initializer == null)
            {
                // If dtype is DT_FLOAT, provide a uniform unit scaling initializer
                if (dtype.is_floating())
                    initializer = tf.glorot_uniform_initializer;
                else if (dtype.is_integer() || dtype.is_unsigned() || dtype.is_bool())
                    initializer = tf.zeros_initializer;
                else if(getter is null)
                    throw new ValueError($"An initializer for variable {name} of type {dtype.as_base_dtype()} is required for layer {name}");
            }

            if (synchronization == VariableSynchronization.OnRead)
                trainable = false;

            var args = new VariableArgs
            {
                Name = name,
                Shape = shape,
                DType = dtype,
                Getter = getter ?? base_layer_utils.make_variable,
                Overwrite = true,
                Initializer = initializer,
                Synchronization = synchronization,
                Aggregation = aggregation,
                Trainable = trainable
            };
            var variable = _add_variable_with_custom_getter(args);

            if (regularizer != null)
            {
                var name_in_scope = variable.Name.Split(':')[0];
                _handle_weight_regularization(name_in_scope, variable, regularizer);
            }

            //backend.track_variable(variable);
            if (trainable == true)
                _trainable_weights.Add(variable);
            else
                _non_trainable_weights.Add(variable);

            return variable;
        }
    }
}
