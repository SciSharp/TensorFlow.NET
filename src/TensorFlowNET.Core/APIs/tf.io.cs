using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class tf
    {
        public static Tensor read_file(string filename, string name = "") => gen_io_ops.read_file(filename, name);

        public static gen_image_ops image => new gen_image_ops();

        public static void import_graph_def(GraphDef graph_def,
            Dictionary<string, Tensor> input_map = null,
            string[] return_elements = null,
            string name = "",
            OpList producer_op_list = null) => importer.import_graph_def(graph_def, input_map, return_elements, name, producer_op_list);
    }
}
