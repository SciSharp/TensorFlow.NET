using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
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
        public VariablePolicy experimental_variable_policy { get; set; } = VariablePolicy.None;
        public bool experimental_custom_gradients { get; set; } = true;
        public SaveOptions(bool save_debug_info = false)
        {
            this.save_debug_info = save_debug_info;
        }
    }

    public class VariablePolicy
    {
        public string Policy { get; }
        private VariablePolicy(string policy)
        {
            Policy = policy;
        }
        public static VariablePolicy None = new(null);
        public static VariablePolicy SAVE_VARIABLE_DEVICES = new("save_variable_devices");
        public static VariablePolicy EXPAND_DISTRIBUTED_VARIABLES = new("expand_distributed_variables");

        public bool save_variable_devices()
        {
            return this != None;
        }

        /// <summary>
        /// Tries to convert `obj` to a VariablePolicy instance.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public static VariablePolicy from_obj(object obj)
        {
            if (obj is null) return None;
            if (obj is VariablePolicy) return (VariablePolicy)obj;
            var key = obj.ToString().ToLower();
            return key switch
            {
                null => None,
                "save_variable_devices" => SAVE_VARIABLE_DEVICES,
                "expand_distributed_variables" => EXPAND_DISTRIBUTED_VARIABLES,
                _ => throw new ValueError($"Received invalid VariablePolicy value: {obj}.")
            };
        }
    }
}
