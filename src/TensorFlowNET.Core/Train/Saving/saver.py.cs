using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class saver
    {
        public static (Saver, object) _import_meta_graph_with_return_elements(string meta_graph_or_file,
            bool clear_devices = false,
            string import_scope = "",
            string[] return_elements = null)
        {
            var meta_graph_def = meta_graph.read_meta_graph_file(meta_graph_or_file);

            var imported_vars = meta_graph.import_scoped_meta_graph_with_return_elements(
                        meta_graph_def,
                        clear_devices: clear_devices,
                        import_scope: import_scope,
                        return_elements: return_elements);

            var saver = _create_saver_from_imported_meta_graph(
                meta_graph_def, import_scope, imported_vars);

            return (saver, null);
        }

        public static Saver _create_saver_from_imported_meta_graph(MetaGraphDef meta_graph_def, 
            string import_scope, 
            (Dictionary<string, RefVariable>, ITensorOrOperation[]) imported_vars)
        {
            if(meta_graph_def.SaverDef != null)
            {
                throw new NotImplementedException("_create_saver_from_imported_meta_graph");
            }
            else
            {
                if(variables._all_saveable_objects(scope: import_scope).Length > 0)
                {
                    // Return the default saver instance for all graph variables.
                    return new Saver();
                }
                else
                {
                    // If no graph variables exist, then a Saver cannot be constructed.
                    Console.WriteLine("Saver not created because there are no variables in the" +
                        " graph to restore");
                    return null;
                }
            }
        }
    }
}
