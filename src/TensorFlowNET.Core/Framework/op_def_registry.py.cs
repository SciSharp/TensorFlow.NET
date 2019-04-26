using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.OpDef.Types;

namespace Tensorflow
{
    public class op_def_registry
    {
        private static Dictionary<string, OpDef> _registered_ops;

        public static Dictionary<string, OpDef> get_registered_ops()
        {
            if(_registered_ops == null)
            {
                _registered_ops = new Dictionary<string, OpDef>();
                var handle = c_api.TF_GetAllOpList();
                var buffer = new Buffer(handle);
                var op_list = OpList.Parser.ParseFrom(buffer);

                foreach (var op_def in op_list.Op)
                    _registered_ops[op_def.Name] = op_def;

                if (!_registered_ops.ContainsKey("NearestNeighbors"))
                    _registered_ops["NearestNeighbors"] = op_NearestNeighbors();
            }

            return _registered_ops;
        }

        /// <summary>
        /// Doesn't work because the op can't be found on binary
        /// </summary>
        /// <returns></returns>
        private static OpDef op_NearestNeighbors()
        {
            var def = new OpDef
            {
                Name = "NearestNeighbors"
            };

            def.InputArg.Add(new ArgDef { Name = "points", Type = DataType.DtFloat });
            def.InputArg.Add(new ArgDef { Name = "centers", Type = DataType.DtFloat });
            def.InputArg.Add(new ArgDef { Name = "k", Type = DataType.DtInt64 });
            def.OutputArg.Add(new ArgDef { Name = "nearest_center_indices", Type = DataType.DtInt64 });
            def.OutputArg.Add(new ArgDef { Name = "nearest_center_distances", Type = DataType.DtFloat });

            return def;
        }
    }
}
