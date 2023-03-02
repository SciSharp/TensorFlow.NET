using System;
using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        public virtual List<ILayer> Layers => _self_tracked_trackables;

        protected void StackLayers(params ILayer[] layers)
        {
            _self_tracked_trackables.AddRange(layers);
        }

        public virtual Shape ComputeOutputShape(Shape input_shape)
            => throw new NotImplementedException("");

        protected List<IVariableV1> _gather_children_variables(bool include_trainable = false, bool include_non_trainable = false)
        {
            List<IVariableV1> res = new();
            var nested_layers = _flatten_layers(false, false);
            foreach (var layer in nested_layers)
            {
                if (layer is Layer l)
                {
                    if (include_trainable == true && include_non_trainable == true)
                    {
                        res.AddRange(l.Variables);
                    }
                    else if (include_trainable == true && include_non_trainable == false)
                    {
                        res.AddRange(l.TrainableVariables);
                    }
                    else if(include_trainable == false && include_non_trainable == true)
                    {
                        res.AddRange(l.NonTrainableVariables);
                    }
                }
            }
            return res;
        }
    }
}
