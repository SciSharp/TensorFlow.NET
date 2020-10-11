using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine
{
    public class TensorFlowOpLayer : Layer
    {
        TensorFlowOpLayerArgs args;
        string _TF_OP_LAYER_NAME_PREFIX = "";

        public TensorFlowOpLayer(TensorFlowOpLayerArgs args) 
            : base(new LayerArgs 
                { 
                    Name = "tf_op_layer_" + args.Name,
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
