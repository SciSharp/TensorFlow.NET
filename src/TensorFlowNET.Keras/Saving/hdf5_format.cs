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
using Tensorflow.Util;
namespace Tensorflow.Keras.Saving
{
    public class hdf5_format
    {
        private static int HDF5_OBJECT_HEADER_LIMIT = 64512;
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

        public static List<(IVariableV1, NDArray)> load_weights_from_hdf5_group(long f, List<ILayer> layers)
        {
            string original_keras_version = "2.5.0";
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

            var filtered_layers = new List<ILayer>();
            foreach (var layer in layers)
            {
                var weights = _legacy_weights(layer);
                if (weights.Count > 0)
                    filtered_layers.append(layer);
            }

            string[] layer_names = load_attributes_from_hdf5_group(f, "layer_names");
            var filtered_layer_names = new List<string>();
            foreach(var name in layer_names)
            {
                if (!filtered_layers.Select(x => x.Name).Contains(name))
                    continue;
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
            return weight_value_tuples;
        }

        public static void toarrayf4(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }

        public static void load_weights_from_hdf5_group_by_name(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }

        public static void save_weights_to_hdf5_group(long f, List<ILayer> layers)
        {
            List<string> layerName=new List<string>();
            foreach (var layer in layers)
            {
                layerName.Add(layer.Name);
            }
            save_attributes_to_hdf5_group(f, "layer_names", layerName.ToArray());
            Hdf5.WriteAttribute(f, "backend", "tensorflow");
            Hdf5.WriteAttribute(f, "keras_version", "2.5.0");

            long g = 0, crDataGroup=0;
            foreach (var layer in layers)
            {
                var weights = _legacy_weights(layer);
                if (weights.Count == 0)
                    continue;

                var weight_names = new List<string>();
                // weight_values= keras.backend.batch_get_value(weights);
                foreach (var weight in weights)
                    weight_names.Add(weight.Name);
                
                g = Hdf5.CreateOrOpenGroup(f, Hdf5Utils.NormalizedName(layer.Name));
                save_attributes_to_hdf5_group(g, "weight_names", weight_names.ToArray());
                foreach (var (name, val) in zip(weight_names, weights))
                {
                    var tensor = val.AsTensor();
                    if (name.IndexOf("/") > 1)
                    {
                        crDataGroup = Hdf5.CreateOrOpenGroup(g, Hdf5Utils.NormalizedName(name.Split('/')[0]));
                        WriteDataset(crDataGroup, name.Split('/')[1], tensor);
                        Hdf5.CloseGroup(crDataGroup);
                    }
                    else
                    {
                        WriteDataset(crDataGroup, name, tensor);
                    }
                }
                Hdf5.CloseGroup(g);
            }
        }

        private static void save_attributes_to_hdf5_group(long f,string name ,Array data)
        {
            int num_chunks = 1;
           
            var chunked_data = Split(data, num_chunks);
            int getSize= 0;
           
            string getType = data.Length>0?data.GetValue(0).GetType().Name.ToLower():"string";

            switch (getType)
            {
                case "single":
                    getSize=sizeof(float);
                    break;
                case "double":
                    getSize = sizeof(double);
                    break;
                case "string":
                    getSize = -1;
                    break;
                case "int32":
                    getSize = sizeof(int);
                    break;
                case "int64":
                    getSize = sizeof(long);
                    break;
                default:
                    getSize=-1;
                    break;
            }
            int getCount = chunked_data.Count;
       
            if (getSize != -1) {
                num_chunks = (int)Math.Ceiling((double)(getCount * getSize) / (double)HDF5_OBJECT_HEADER_LIMIT);
                if (num_chunks > 1) chunked_data = Split(data, num_chunks);
            }
            
            if (num_chunks > 1)
            {
                foreach (var (chunk_id, chunk_data) in enumerate(chunked_data))
                {
                    
                    WriteAttrs(f, getType, $"{name}{chunk_id}", chunk_data.ToArray());
              
                }

            }
            else {
        
                WriteAttrs(f, getType,name, data);
              
            }
        }

        private static void WriteDataset(long f,  string name, Tensor data)
        {
            switch (data.dtype)
            {
                case TF_DataType.TF_FLOAT:
                    Hdf5.WriteDatasetFromArray<float>(f, name, data.numpy().ToMuliDimArray<float>());
                    break;
                case TF_DataType.TF_DOUBLE:
                    Hdf5.WriteDatasetFromArray<double>(f, name, data.numpy().ToMuliDimArray<double>());
                    break;
                case TF_DataType.TF_INT32:
                    Hdf5.WriteDatasetFromArray<int>(f, name, data.numpy().ToMuliDimArray<int>());
                    break;
                case TF_DataType.TF_INT64:
                    Hdf5.WriteDatasetFromArray<long>(f, name, data.numpy().ToMuliDimArray<long>());
                    break;
                default:
                    Hdf5.WriteDatasetFromArray<float>(f, name, data.numpy().ToMuliDimArray<float>());
                    break;
            }
        }

        private static void WriteAttrs(long f,string typename, string name, Array data)
        {
            switch (typename)
            {
                case "single":
                    Hdf5.WriteAttributes<float>(f, name, data);
                    break;
                case "double":
                    Hdf5.WriteAttributes<double>(f, name, data);
                    break;
                case "string":
                    Hdf5.WriteAttributes<string>(f, name, data);
                    break;
                case "int32":
                    Hdf5.WriteAttributes<int>(f, name, data);
                    break;
                case "int64":
                    Hdf5.WriteAttributes<long>(f, name, data);
                    break;
                default:
                    Hdf5.WriteAttributes<string>(f, name,data);
                    break;
            }
        }

        private static List<List<object>> Split(Array list, int chunkSize)
        {
            var splitList = new List<List<object>>();
            var chunkCount = (int)Math.Ceiling((double)list.Length / (double)chunkSize);

            for (int c = 0; c < chunkCount; c++)
            {
                var skip = c * chunkSize;
                var take = skip + chunkSize;
                var chunk = new List<object>(chunkSize);

                for (int e = skip; e < take && e < list.Length; e++)
                {
                    chunk.Add(list.GetValue(e));
                }
                splitList.Add(chunk);
            }

            return splitList;
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

