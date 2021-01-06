using System;
using System.Collections.Generic;
using System.Text;
using HDF.PInvoke;
using NumSharp;
using Tensorflow.Keras.Engine;
using HDF5CSharp;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
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
        public static void preprocess_weights_for_loading(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }
        public static void _convert_rnn_weights(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

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
        public static void load_weights_from_hdf5_group(long f=-1,Model model=null)
        {
            string original_keras_version = "1";
            string original_backend = null;
            if (Hdf5.AttributeExists(f, "keras_version"))
            {
                (bool success, string[] attr) = Hdf5.ReadStringAttributes(f, "keras_version", "");
                if (success)
                {
                    original_keras_version = attr[0];
                }
            }
            if (Hdf5.AttributeExists(f, "backend"))
            {
                (bool success, string[] attr) = Hdf5.ReadStringAttributes(f, "backend", "");
                if (success)
                {
                    original_backend = attr[0];
                }
            }
            List<ILayer> filtered_layers = new List<ILayer>();
            List<Tensor> weights;
            foreach (var layer in model.Layers)
            {
                weights = _legacy_weights(layer);
                if (weights.Count>0)
                {
                    filtered_layers.append(layer);
                }
            }
            string[] layer_names = load_attributes_from_hdf5_group(f,"layer_names");
            List<NDArray> weight_values=new List<NDArray>();
            foreach (var i in filtered_layers) { 
                long g = H5G.open(f, i.Name);
                string[] weight_names = null;
                if (g != -1)
                {
                    weight_names = load_attributes_from_hdf5_group(g, "weight_names");
                }
                if (weight_names != null)
                {
                    foreach (var i_ in weight_names) {
                        (bool success, Array result) = Hdf5.ReadDataset<float>(g, i_);
                        //
                        weight_values.Add(np.array(result));
                    }
                }
                H5G.close(g);
            }

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
        public static string[] load_attributes_from_hdf5_group(long f = -1, string name = "")
        {
            if (Hdf5.AttributeExists(f, name))
            {
                (bool success, string[] attr) = Hdf5.ReadStringAttributes(f, name, "");
                if (success)
                {
                    return attr;
                }
            }
            return null;
        }
        public static void load_attributes_from_hdf5_group(long filepath = -1, Dictionary<string, object> custom_objects = null, bool compile = false)
        {

        }

        public static List<Tensor> _legacy_weights(ILayer layer)
        {
        
            List<Tensor> weights= new List<Tensor>();
            if (layer.trainable_weights.Count != 0)
            {
                Tensor[] trainable_weights = Array.ConvertAll<IVariableV1, Tensor>(layer.trainable_weights.ToArray(), s => s.AsTensor());
                Tensor[] non_trainable_weights =null;
                if (layer.non_trainable_weights.Count != 0)
                {
                    non_trainable_weights = Array.ConvertAll<IVariableV1, Tensor>(layer.non_trainable_weights.ToArray(), s => s.AsTensor());
                }
                foreach (var i in trainable_weights) {
                    if (non_trainable_weights != null)
                    {
                        foreach (var i_ in non_trainable_weights)
                        {
                            weights.Add(i + i_);
                        }
                    }
                    else {
                        weights.Add(i);
                    };

                   
                }
            }
            return weights;
        }
    }
}

