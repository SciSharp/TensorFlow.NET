using System;
using System.Collections.Generic;
using System.Text;
using HDF.PInvoke;
using HDF5CSharp;
using NumSharp;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Engine
{
     public partial class Model
    {
        public List<(IVariableV1, NDArray)> load_weights(string filepath, bool by_name = false, bool skip_mismatch = false, object options = null)
        {
            long fileId = Hdf5.OpenFile(filepath, true);

            bool msuccess = Hdf5.GroupExists(fileId, "model_weights");
            bool lsuccess = Hdf5.GroupExists(fileId, "layer_names");

            if (!lsuccess && msuccess)
                fileId = H5G.open(fileId, "model_weights");

            if (by_name)
                //fdf5_format.load_weights_from_hdf5_group_by_name();
                throw new NotImplementedException("");
            else
            {
                var weights = hdf5_format.load_weights_from_hdf5_group(fileId, Layers);
                Hdf5.CloseFile(fileId);
                // return a reference to prevent GC collect Variable.
                return weights;
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

