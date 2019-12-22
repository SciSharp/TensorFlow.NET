using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class GraphTransformer
    {
        /// <summary>
        /// Graph Transform Tool
        /// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
        /// </summary>
        /// <param name="input_graph_def">GraphDef object containing a model to be transformed</param>
        /// <param name="inputs">the model inputs</param>
        /// <param name="outputs">the model outputs</param>
        /// <param name="transforms">transform names and parameters</param>
        /// <returns></returns>
        public GraphDef TransformGraph(GraphDef input_graph_def, 
            string[] inputs, 
            string[] outputs, 
            string[] transforms)
        {
            var input_graph_def_string = input_graph_def.ToString();
            var inputs_string = string.Join(",", inputs);
            var outputs_string = string.Join(",", outputs);
            var transforms_string = string.Join(",", transforms);

            throw new NotImplementedException("");
        }
    }
}
