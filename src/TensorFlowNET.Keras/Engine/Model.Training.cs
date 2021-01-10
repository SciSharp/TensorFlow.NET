using System;
using System.Collections.Generic;
using System.Text;
using HDF.PInvoke;
using HDF5CSharp;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Engine
{
     public partial class Model
    {
        public void load_weights(string filepath, bool by_name = false, bool skip_mismatch = false, object options = null)
        {
            long fileId = Hdf5.OpenFile(filepath, true);

            bool msuccess = Hdf5.GroupExists(fileId, "model_weights");
            bool lsuccess = Hdf5.GroupExists(fileId, "layer_names");

            if (!lsuccess && msuccess)
            {
                fileId = H5G.open(fileId, "model_weights");
            }
            if (by_name)
            {
                //fdf5_format.load_weights_from_hdf5_group_by_name();
                throw new NotImplementedException("");
            }
            else
            {
                fdf5_format.load_weights_from_hdf5_group(fileId, Layers);
            }
            H5G.close(fileId);
        }
    }
}

