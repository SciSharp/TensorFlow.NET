using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class graph_util_impl
    {
        /// <summary>
        /// Replaces all the variables in a graph with constants of the same values.
        /// </summary>
        /// <param name="sess">Active TensorFlow session containing the variables.</param>
        /// <param name="input_graph_def">GraphDef object holding the network.</param>
        /// <param name="output_node_names">List of name strings for the result nodes of the graph.</param>
        /// <param name="variable_names_whitelist"></param>
        /// <param name="variable_names_blacklist"></param>
        /// <returns>GraphDef containing a simplified version of the original.</returns>
        public GraphDef convert_variables_to_constants(Session sess,
                                   GraphDef input_graph_def,
                                   string[] output_node_names,
                                   string[] variable_names_whitelist = null,
                                   string[] variable_names_blacklist = null)
        {
            // This graph only includes the nodes needed to evaluate the output nodes, and
            // removes unneeded nodes like those involved in saving and assignment.
            throw new NotImplementedException("");
        }

        private string get_input_name(string node)
        {
            throw new NotImplementedException("");
        }
    }
}
