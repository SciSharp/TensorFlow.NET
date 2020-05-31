using Keras.Layers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras
{
    class Models
    {
        public class Model : Keras.Engine.Training.Model{}

        public static Layer share_weights(Layer layer) => throw new NotImplementedException();

        private static Layer _clone_layer(Layer layer) => throw new NotImplementedException();

        private static Layer _insert_ancillary_layers(Model model, Layer ancillary_layers, string[] metrics_names, Node[] new_nodes) => throw new NotImplementedException();

        private static Node[] _make_new_nodes(Node[] nodes_by_depth, Func<Layer, Layer> layer_fn, Hashtable layer_map, Hashtable tensor_map) => throw new NotImplementedException();

        private static Model _clone_functional_model(Model model, Tensor[] input_tensors = null, Func<Layer, Layer> layer_fn = null) => throw new NotImplementedException();

        private static (Hashtable, Layer[]) _clone_layers_and_model_config(Model model, Layer[] input_layers, Func<Layer, Layer> layer_fn) => throw new NotImplementedException();

        private static (Layer[], Layer[]) _remove_ancillary_layers(Model model, Hashtable layer_map, Layer[] layers) => throw new NotImplementedException();

        private static Sequential _clone_sequential_model(Model model, Tensor[] input_tensors = null, Func<Layer, Layer> layer_fn = null) => throw new NotImplementedException();

        public static Model clone_model(Model model, Tensor[] input_tensors = null, Func<Layer, Layer> layer_fn = null) => throw new NotImplementedException();

        private static void _in_place_subclassed_model_reset(Model model) => throw new NotImplementedException();

        private static void _reset_build_compile_trackers(Model model) => throw new NotImplementedException();

        public static void in_place_subclassed_model_state_restoration(Model model) => throw new NotImplementedException();

        public static void clone_and_build_model(Model model, Tensor[] input_tensors= null, Tensor[] target_tensors= null, object custom_objects= null,
                                        bool compile_clone= true, bool in_place_reset= false, IVariableV1 optimizer_iterations= null, Hashtable optimizer_config= null)
            => throw new NotImplementedException();
    }
}
