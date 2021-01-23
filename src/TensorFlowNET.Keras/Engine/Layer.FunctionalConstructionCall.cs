using System;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        Tensors FunctionalConstructionCall(Tensors inputs)
        {
            bool mask_arg_passed_by_framework = false;
            bool training_arg_passed_by_framework = false;
            Tensor training_value = null;
            if (training_value == null)
            {
                training_arg_passed_by_framework = true;
            }

            if (base_layer_utils.needs_keras_history(inputs))
                base_layer_utils.create_keras_history(inputs);

            Tensors outputs = null;
            using var ctxManager = CallContext.enter(build_graph: true);

            var graph = keras.backend.get_graph();
            graph.as_default();

            tf_with(ops.name_scope(_name_scope()), scope =>
            {
                MaybeBuild(inputs);

                // Wrapping `call` function in autograph to allow for dynamic control
                // flow and control dependencies in call. We are limiting this to
                // subclassed layers as autograph is strictly needed only for
                // subclassed layers and models.
                // tf_convert will respect the value of autograph setting in the
                // enclosing tf.function, if any.
                if (!dynamic)
                    throw new NotImplementedException("");

                outputs = Call(inputs);
                
                _set_connectivity_metadata_(inputs, outputs);
                _handle_activity_regularization(inputs, outputs);
                _set_mask_metadata(inputs, outputs, null);
            });

            graph.Exit();

            return outputs;
        }
    }
}
