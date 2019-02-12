using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class saver
    {
        public static Saver _import_meta_graph_with_return_elements(string meta_graph_or_file,
            bool clear_devices = false,
            string import_scope = "",
            string[] return_elements = null)
        {
            var meta_graph_def = meta_graph.read_meta_graph_file(meta_graph_or_file);

            meta_graph.import_scoped_meta_graph_with_return_elements(
                        meta_graph_def,
                        clear_devices: clear_devices,
                        import_scope: import_scope,
                        return_elements: return_elements);

            return null;
            /*var (imported_vars, imported_return_elements) = (
                    , false);*/
        }
    }
}
