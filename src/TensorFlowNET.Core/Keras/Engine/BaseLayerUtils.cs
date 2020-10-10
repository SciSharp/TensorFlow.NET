using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public class BaseLayerUtils
    {
        public static Layer[] CreateKerasHistoryHelper(Tensors tensors)
        {
            var processed_ops = new List<Operation>();
            var created_layers = new List<Layer>();

            foreach (var tensor in tensors)
            {
                if (tensor.KerasHistory != null)
                    continue;

                var op = tensor.op;
                if (!processed_ops.Contains(op))
                {
                    var layer_inputs = new List<Tensor>();

                    foreach (var (i, op_input) in enumerate(op.inputs._inputs))
                    {
                        if (uses_keras_history(op_input))
                            layer_inputs.Add(op_input);
                        else
                        {

                        }
                    }
                }
            }

            return created_layers.ToArray();
        }

        static bool uses_keras_history(Tensor op_input)
        {
            return Layer.KerasHistories.Any(x => x.tensor == op_input);
        }
    }
}
