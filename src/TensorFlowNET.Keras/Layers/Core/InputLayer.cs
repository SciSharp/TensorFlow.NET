/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System.Linq;
using Tensorflow.Framework.Models;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Layer to be used as an entry point into a Network (a graph of layers).
    /// </summary>
    public class InputLayer : Layer
    {
        InputLayerArgs args;
        bool isPlaceholder;
        TensorSpec typeSpec;

        public InputLayer(InputLayerArgs args) :
            base(args)
        {
            this.args = args;
            built = true;
            SupportsMasking = true;

            if (BatchInputShape != null)
            {
                args.BatchSize = BatchInputShape.dims[0];
                args.InputShape = BatchInputShape.dims.Skip(1).ToArray();
            }

            // moved to base class
            if (string.IsNullOrEmpty(args.Name))
            {
                var prefix = "input";
                name = prefix + '_' + keras.backend.get_uid(prefix);
                args.Name = name;
            }

            if (args.DType == TF_DataType.DtInvalid)
            {
                args.DType = args.InputTensor == null ? tf.float32 : args.InputTensor.dtype;
            }

            if (args.InputTensor == null)
            {
                if (args.InputShape != null)
                {
                    args.BatchInputShape = new int[] { args.BatchSize }
                        .Concat(args.InputShape.dims)
                        .ToArray();
                }
                else
                {
                    args.BatchInputShape = null;
                }

                var graph = keras.backend.get_graph();
                graph.as_default();

                args.InputTensor = keras.backend.placeholder(
                    shape: BatchInputShape,
                    dtype: DType,
                    name: Name,
                    sparse: args.Sparse,
                    ragged: args.Ragged);

                graph.Exit();

                isPlaceholder = true;
            }

            // Create an input node to add to self.outbound_node
            // and set output_tensors' _keras_history.
            // input_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
            // input_tensor._keras_mask = None
            var node = new Node(new NodeArgs
            {
                Outputs = args.InputTensor
            });
            node.Connect(this);

            typeSpec = new TensorSpec(args.InputTensor.TensorShape,
                dtype: args.InputTensor.dtype,
                name: Name);
        }

        public static InputLayer from_config(LayerArgs args)
        {
            return new InputLayer(args as InputLayerArgs);
        }
    }
}
