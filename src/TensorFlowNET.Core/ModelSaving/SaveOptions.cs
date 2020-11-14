using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.ModelSaving
{
    /// <summary>
    /// Options for saving to SavedModel.
    /// </summary>
    public class SaveOptions
    {
        bool save_debug_info;
        public SaveOptions(bool save_debug_info = false)
        {
            this.save_debug_info = save_debug_info;
        }
    }
}
