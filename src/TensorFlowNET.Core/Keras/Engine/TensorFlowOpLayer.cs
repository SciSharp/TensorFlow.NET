using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine
{
    public class TensorFlowOpLayer : Layer
    {
        TensorFlowOpLayerArgs args;
        static string TF_OP_LAYER_NAME_PREFIX = "tf_op_layer_";

        public TensorFlowOpLayer(TensorFlowOpLayerArgs args) 
            : base(new LayerArgs 
                { 
                    Name = TF_OP_LAYER_NAME_PREFIX + args.Name,
                    Trainable = args.Trainable,
                    DType = args.DType,
                    Autocast = false
                })
        {
            this.args = args;
            built = true;
        }

        protected override Tensors call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            return base.call(inputs, state, is_training);
        }
    }
}
