using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class variables
    {
        /// <summary>
        /// Returns all variables created with `trainable=True`
        /// </summary>
        /// <returns></returns>
        public static object trainable_variables()
        {
            return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES);
        }

        /// <summary>
        /// Returns global variables.
        /// </summary>
        /// <param name="scope">
        /// (Optional.) A string. If supplied, the resulting list is filtered
        /// to include only items whose `name` attribute matches `scope` using
        /// `re.match`. Items without a `name` attribute are never returned if a
        /// scope is supplied. The choice of `re.match` means that a `scope` without
        /// special tokens filters by prefix.
        /// </param>
        /// <returns>A list of `Variable` objects.</returns>
        public static List<RefVariable> global_variables(string scope = "")
        {
            var result = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope);

            return result as List<RefVariable>;
        }

        /// <summary>
        /// Returns an Op that initializes a list of variables.
        /// </summary>
        /// <param name="var_list">List of `Variable` objects to initialize.</param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <returns>An Op that run the initializers of all the specified variables.</returns>
        public static Operation variables_initializer(RefVariable[] var_list, string name = "init")
        {
            return control_flow_ops.group(var_list.Select(x => x.initializer).ToArray(), name);
        }
    }
}
