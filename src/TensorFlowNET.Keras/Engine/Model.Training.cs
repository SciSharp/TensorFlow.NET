using System;
using System.Collections.Generic;
using System.Text;
using HDF.PInvoke;
using HDF5CSharp;
using static Tensorflow.Binding;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        static Dictionary<string, List<(string, NDArray)>> weightsCache
            = new Dictionary<string, List<(string, NDArray)>>();

        public void load_weights(string filepath, bool by_name = false, bool skip_mismatch = false, object options = null)
        {
            // Get from cache
            if (weightsCache.ContainsKey(filepath))
            {
                var filtered_layers = new List<ILayer>();
                foreach (var layer in Layers)
                {
                    var weights = hdf5_format._legacy_weights(layer);
                    if (weights.Count > 0)
                        filtered_layers.append(layer);
                }

                var weight_value_tuples = new List<(IVariableV1, NDArray)>();
                filtered_layers.Select((layer, i) =>
                {
                    var symbolic_weights = hdf5_format._legacy_weights(layer);
                    foreach(var weight in symbolic_weights)
                    {
                        var weight_value = weightsCache[filepath].First(x => x.Item1 == weight.Name).Item2;
                        weight_value_tuples.Add((weight, weight_value));
                    }
                    return layer;
                }).ToList();

                keras.backend.batch_set_value(weight_value_tuples);
                return;
            }

            long fileId = Hdf5.OpenFile(filepath, true);
            if(fileId < 0)
            {
                tf_output_redirect.WriteLine($"Can't find weights file {filepath}");
                return;
            }
            bool msuccess = Hdf5.GroupExists(fileId, "model_weights");
            bool lsuccess = Hdf5.GroupExists(fileId, "layer_names");

            if (!lsuccess && msuccess)
                fileId = H5G.open(fileId, "model_weights");

            if (by_name)
                //fdf5_format.load_weights_from_hdf5_group_by_name();
                throw new NotImplementedException("");
            else
            {
                var weight_value_tuples = hdf5_format.load_weights_from_hdf5_group(fileId, Layers);
                Hdf5.CloseFile(fileId);

                weightsCache[filepath] = weight_value_tuples.Select(x => (x.Item1.Name, x.Item2)).ToList();
                keras.backend.batch_set_value(weight_value_tuples);
            }
        }

        public void save_weights(string filepath, bool overwrite = true, string save_format = null, object options = null)
        {
            long fileId = Hdf5.CreateFile(filepath);
            hdf5_format.save_weights_to_hdf5_group(fileId, Layers);
            Hdf5.CloseFile(fileId);
        }
    }
}

