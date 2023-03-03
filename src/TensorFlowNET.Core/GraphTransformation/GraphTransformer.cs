using Google.Protobuf;

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
            var input_graph_def_string = input_graph_def.ToByteArray();
            var inputs_string = string.Join(",", inputs);
            var outputs_string = string.Join(",", outputs);
            var transforms_string = string.Join(" ", transforms);
            var status = new Status();
            var buffer = new Buffer();
            var len = c_api.TransformGraphWithStringInputs(input_graph_def_string,
                input_graph_def_string.Length,
                inputs_string,
                outputs_string,
                transforms_string,
                buffer,
                status);

            status.Check(false);
            var bytes = buffer.ToArray();
            return GraphDef.Parser.ParseFrom(bytes);
        }
    }
}
