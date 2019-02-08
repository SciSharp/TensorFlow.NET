using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class BaseSaverBuilder
    {
        protected int _write_version;

        public BaseSaverBuilder(int write_version = 2)
        {
            _write_version = write_version;
        }

        public virtual Operation save_op(Tensor filename_tensor, SaveableObject[] saveables)
        {
            var tensor_names = new List<string>();
            var tensors = new List<Tensor>();
            var tensor_slices = new List<string>();

            foreach (var saveable in saveables)
            {
                foreach(var spec in saveable.specs)
                {
                    tensor_names.Add(spec.name);
                    tensors.Add(spec.tensor);
                    tensor_slices.Add(spec.slice_spec);
                }
            }

            if (_write_version == 2)
            {
                return gen_io_ops.save_v2(filename_tensor, tensor_names.ToArray(), tensor_slices.ToArray(), tensors.ToArray());
            }
            else
            {
                throw new NotImplementedException("_write_version v1");
            }
        }

        public virtual Tensor[] bulk_restore(Tensor filename_tensor, SaveableObject[] saveables, int preferred_shard, bool restore_sequentially)
        {
            throw new NotImplementedException();
        }

        public virtual SaverDef _build_internal(RefVariable[] names_to_saveables,
            bool reshape = false,
            bool sharded = false,
            int max_to_keep = 5,
            double keep_checkpoint_every_n_hours = 10000,
            string name = "",
            bool restore_sequentially = false,
            string filename = "model",
            bool build_save = true,
            bool build_restore = true)
        {
            if (!build_save || !build_restore)
                throw new ValueError("save and restore operations need to be built together " +
                    " when eager execution is not enabled.");

            var saveables = saveable_object_util.validate_and_slice_inputs(names_to_saveables);

            if (max_to_keep < 0)
                max_to_keep = 0;

            Python.with<ops.name_scope>(new ops.name_scope(name, "save", saveables.Select(x => x.op).ToArray()), scope =>
            {
                name = scope;

                // Add a placeholder string tensor for the filename.
                var filename_tensor = gen_array_ops.placeholder_with_default( string.IsNullOrEmpty(filename) ? "model" : filename, shape: new TensorShape(), name: "filename");
                filename_tensor = gen_array_ops.placeholder_with_default(filename_tensor, shape: new TensorShape(), name: "Const");
                // Keep the name "Const" for backwards compatibility.

                // Add the save ops.
                if (sharded)
                {

                }
                else
                {
                    if (build_save)
                        _AddSaveOps(filename_tensor, saveables);
                }
            });

            throw new NotImplementedException("");
        }

        public Tensor _AddSaveOps(Tensor filename_tensor, SaveableObject[] saveables)
        {
            var save = save_op(filename_tensor, saveables);
            return control_flow_ops.with_dependencies(new Operation[] { save }, filename_tensor);
        }
    }
}
