using System;
using System.Collections.Generic;
using System.Text;
using HDF.PInvoke;
using NumSharp;
using Tensorflow.Keras.Engine;
using HDF5CSharp;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.Linq;

namespace Tensorflow.Keras.Saving
{
    public class fdf5_format
    {

        public static void load_model_from_hdf5(string filepath = "", Dictionary<string, object> custom_objects = null, bool compile = false)
        {
            long root = Hdf5.OpenFile(filepath,true);
            load_model_from_hdf5(root, custom_objects, compile);
        }
        public static void load_model_from_hdf5(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {
            //long fileId = filepath;
            //try
            //{
            //    groupId = H5G.open(fileId, "/");
            //    (bool success, string[] attrId) = Hdf5.ReadStringAttributes(groupId, "model_config", "");
            //    H5G.close(groupId);
            //    if (success == true) {
            //        Console.WriteLine(attrId[0]);
            //    }
            //}
            //catch (Exception ex)
            //{
            //    if (filepath != -1) {
            //        Hdf5.CloseFile(filepath);
            //    }
            //    if (groupId != -1) {
            //        H5G.close(groupId);
            //    }
            //    throw new Exception(ex.ToString());
            //}

        }
        public static void save_model_to_hdf5(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }

        /// <summary>
        /// Preprocess layer weights between different Keras formats.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="weights"></param>
        /// <param name="original_keras_version"></param>
        /// <param name="original_backend"></param>
        public static List<NDArray> preprocess_weights_for_loading(ILayer layer, List<NDArray> weights, string original_keras_version = null, string original_backend = null)
        {
            // convert CuDNN layers
            return _convert_rnn_weights(layer, weights);
        }

        /// <summary>
        /// Converts weights for RNN layers between native and CuDNN format.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="weights"></param>
        static List<NDArray> _convert_rnn_weights(ILayer layer, List<NDArray> weights)
        {
            var target_class = layer.GetType().Name;
            return weights;
        }
        public static void save_optimizer_weights_to_hdf5_group(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }
        public static void load_optimizer_weights_from_hdf5_group(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }
        public static void save_weights_to_hdf5_group(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }
        public static void load_weights_from_hdf5_group(long f, List<ILayer> layers)
        {
            string original_keras_version = "2.4.0";
            string original_backend = null;
            if (Hdf5.AttributeExists(f, "keras_version"))
            {
                var (success, attr) = Hdf5.ReadStringAttributes(f, "keras_version", "");
                if (success)
                    original_keras_version = attr.First();
                // keras version should be 2.5.0+
                var ver_major = int.Parse(original_keras_version.Split('.')[0]);
                var ver_minor = int.Parse(original_keras_version.Split('.')[1]);
                if (ver_major < 2 || (ver_major == 2 && ver_minor < 5))
                    throw new ValueError("keras version should be 2.5.0 or later.");
            }
            if (Hdf5.AttributeExists(f, "backend"))
            {
                var (success, attr) = Hdf5.ReadStringAttributes(f, "backend", "");
                if (success)
                    original_backend = attr.First();
            }
            List<ILayer> filtered_layers = new List<ILayer>();
            List<IVariableV1> weights;
            foreach (var layer in layers)
            {
                weights = _legacy_weights(layer);
                if (weights.Count > 0)
                {
                    filtered_layers.append(layer);
                }
            }
            string[] layer_names = load_attributes_from_hdf5_group(f, "layer_names");
            var filtered_layer_names = new List<string>();
            foreach(var name in layer_names)
            {
                long g = H5G.open(f, name);
                var weight_names = load_attributes_from_hdf5_group(g, "weight_names");
                if (weight_names.Count() > 0)
                    filtered_layer_names.Add(name);
                H5G.close(g);
            }
            layer_names = filtered_layer_names.ToArray();
            if (layer_names.Length != filtered_layers.Count())
                throw new ValueError("You are trying to load a weight file " +
                    $"containing {layer_names}" +
                    $" layers into a model with {filtered_layers.Count} layers.");

            var weight_value_tuples = new List<(IVariableV1, NDArray)>();
            foreach (var (k, name) in enumerate(layer_names))
            {
                var weight_values = new List<NDArray>();
                long g = H5G.open(f, name);
                var weight_names = load_attributes_from_hdf5_group(g, "weight_names");
                foreach (var i_ in weight_names)
                {
                    (bool success, Array result) = Hdf5.ReadDataset<float>(g, i_);
                    if (success)
                        weight_values.Add(np.array(result));
                }
                H5G.close(g);
                var layer = filtered_layers[k];
                var symbolic_weights = _legacy_weights(layer);
                preprocess_weights_for_loading(layer, weight_values, original_keras_version, original_backend);
                if (weight_values.Count() != symbolic_weights.Count())
                    throw new ValueError($"Layer #{k} (named {layer.Name}" +
                        "in the current model) was found to " +
                        $"correspond to layer {name} in the save file." +
                        $"However the new layer {layer.Name} expects " +
                        $"{symbolic_weights.Count()} weights, but the saved weights have " +
                        $"{weight_values.Count()} elements.");
                weight_value_tuples.AddRange(zip(symbolic_weights, weight_values));
            }
            keras.backend.batch_set_value(weight_value_tuples);
        }
        public static void toarrayf4(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }
        public static void load_weights_from_hdf5_group_by_name(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }
        public static void save_attributes_to_hdf5_group(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }
        public static string[] load_attributes_from_hdf5_group(long group, string name)
        {
            if (Hdf5.AttributeExists(group, name))
            {
                var (success, attr) = Hdf5.ReadStringAttributes(group, name, "");
                if (success)
                    return attr.ToArray();
            }
            return null;
        }
        public static void load_attributes_from_hdf5_group(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }

        public static List<IVariableV1> _legacy_weights(ILayer layer)
        {
            var weights = layer.trainable_weights.Select(x => x).ToList();
            weights.AddRange(layer.non_trainable_weights);
            return weights;
        }
    }
}

