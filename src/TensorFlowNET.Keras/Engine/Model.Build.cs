using System;
using System.Linq;
using Tensorflow.Graphs;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        public override void build(KerasShapesWrapper input_shape)
        {
            if (_is_graph_network || this is Functional || this is Sequential)
            {
                base.build(input_shape);
                return;
            }

            if(input_shape is not null && this.inputs is null)
            {
                var graph = tf.executing_eagerly() ? new FuncGraph("build_graph") : keras.backend.get_graph();
                graph.as_default();
                var shapes = input_shape.ToShapeArray();
                var x = new Tensors(shapes.Select(x => base_layer_utils.generate_placeholders_from_shape(x)).ToArray());
                try
                {
                    Call(x, training: false);
                }
                catch (InvalidArgumentError)
                {
                    throw new ValueError("You cannot build your model by calling `build` " +
                        "if your layers do not support float type inputs. " +
                        "Instead, in order to instantiate and build your " +
                        "model, `call` your model on real tensor data (of the correct dtype).");
                }
                catch (TypeError)
                {
                    throw new ValueError("You cannot build your model by calling `build` " +
                        "if your layers do not support float type inputs. " +
                        "Instead, in order to instantiate and build your " +
                        "model, `call` your model on real tensor data (of the correct dtype).");
                }
                graph.Exit();
            }

            base.build(input_shape);
        }
    }
}
