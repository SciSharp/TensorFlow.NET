using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Train;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public SavedModelAPI saved_model { get; } = new SavedModelAPI();
    }

    public class SavedModelAPI
    {
        public Trackable load(string export_dir, LoadOptions? options = null)
        {
            return Loader.load(export_dir, options);
        }
    }
}
