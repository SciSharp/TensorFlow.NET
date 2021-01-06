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
        private  long fileId = -1;
        private  long f = -1;
        public  void load_weights(string filepath ="",bool by_name= false, bool skip_mismatch=false, object options = null)
        {
            long root = Hdf5.OpenFile(filepath, true);

            long fileId = root;
            //try
            //{

                bool msuccess = Hdf5.GroupExists(fileId, "model_weights");
                bool lsuccess = Hdf5.GroupExists(fileId, "layer_names");

                if (!lsuccess && msuccess)
                {
                    f = H5G.open(fileId, "model_weights");
                   
                }
                if (by_name)
                {
                    //fdf5_format.load_weights_from_hdf5_group_by_name();
                }
                else
                {
                    fdf5_format.load_weights_from_hdf5_group(f, this);
                }
                H5G.close(f);
            //}
            //catch (Exception ex)
            //{
            //    if (fileId != -1)
            //    {
            //        Hdf5.CloseFile(fileId);
            //    }
            //    if (f != -1)
            //    {
            //        H5G.close(f);
            //    }
            //    throw new Exception(ex.ToString());
            //}
        }
   
    }
}

