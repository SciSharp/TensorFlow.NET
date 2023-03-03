using System.Collections.Generic;
using Tensorflow.Functions;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.ModelSaving;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        ModelSaver saver = new ModelSaver();

        /// <summary>
        /// Saves the model to Tensorflow SavedModel or a single HDF5 file.
        /// </summary>
        /// <param name="filepath"></param>
        /// <param name="overwrite"></param>
        /// <param name="include_optimizer"></param>
        public void save(string filepath,
            bool overwrite = true,
            bool include_optimizer = true,
            string save_format = "tf",
            SaveOptions? options = null,
            ConcreteFunction? signatures = null,
            bool save_traces = true)
        {
            if (save_format != "tf")
            {
                saver.save(this, filepath);
            }
            else
            {
                using (SharedObjectSavingScope.Enter())
                {
                    KerasSavedModelUtils.save_model(this, filepath, overwrite, include_optimizer, signatures, options, save_traces);
                }
            }
        }
    }
}
