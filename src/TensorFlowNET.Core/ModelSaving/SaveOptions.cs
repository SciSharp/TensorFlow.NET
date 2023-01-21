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
        public bool save_debug_info = false;
        public IList<string>? namespace_white_list { get; set; } = null;
        public IDictionary<string, object>? function_aliases { get; set; } = null;
        public string? experimental_io_device { get; set; } = null;
        // TODO: experimental
        public Object? experimental_variable_polict { get; set; } = null;
        public bool experimental_custom_gradients { get; set; } = true;
        public SaveOptions(bool save_debug_info = false)
        {
            this.save_debug_info = save_debug_info;
        }
    }
}
