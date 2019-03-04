using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class BaseSaverBuilder : Python
    {
        protected SaverDef.Types.CheckpointFormatVersion _write_version;

        public BaseSaverBuilder(SaverDef.Types.CheckpointFormatVersion write_version = SaverDef.Types.CheckpointFormatVersion.V2)
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

            if (_write_version == SaverDef.Types.CheckpointFormatVersion.V2)
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
            var names = new List<string>();
            var slices = new List<string>();
            var dtypes = new List<TF_DataType>();
            foreach (var saveable in saveables)
                foreach (var spec in saveable.specs)
                {
                    names.Add(spec.name);
                    slices.Add(spec.slice_spec);
                    dtypes.Add(spec.dtype);
                }

            return gen_io_ops.restore_v2(filename_tensor, names.ToArray(), slices.ToArray(), dtypes.ToArray());
        }

        public virtual SaverDef _build_internal(RefVariable[] names_to_saveables,
            bool reshape = false,
            bool sharded = false,
            int max_to_keep = 5,
            float keep_checkpoint_every_n_hours = 10000,
            string name = null,
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

            Tensor save_tensor = null;
            Operation restore_op = null;

            return with(new ops.name_scope(name, "save", saveables.Select(x => x.op).ToArray()), scope =>
            {
                name = scope;

                // Add a placeholder string tensor for the filename.
                var filename_tensor = array_ops.placeholder_with_default(string.IsNullOrEmpty(filename) ? "model" : filename, shape: new int[0], name: "filename");
                // Keep the name "Const" for backwards compatibility.
                filename_tensor = gen_array_ops.placeholder_with_default(filename_tensor, shape: new int[0], name: "Const");

                // Add the save ops.
                if (sharded)
                {

                }
                else
                {
                    if (build_save)
                        save_tensor = _AddSaveOps(filename_tensor, saveables);

                    if (build_restore)
                        restore_op = _AddRestoreOps(filename_tensor, saveables, restore_sequentially, reshape);
                }

                var graph = ops.get_default_graph();
                var check_collection_list = graph.get_all_collection_keys();
                foreach (var collection_type in check_collection_list)
                {
                    var cols = graph.get_collection(collection_type);
                    switch (cols)
                    {
                        case List<RefVariable> values:
                            foreach (var element in values) ;
                            break;
                        case List<ITensorOrOperation> values:
                            foreach (var element in values) ;
                            break;
                        default:
                            throw new NotImplementedException("_build_internal.check_collection_list");
                    }
                    
                }

                return new SaverDef()
                {
                    FilenameTensorName = filename_tensor.name,
                    SaveTensorName = save_tensor.name,
                    RestoreOpName = restore_op.name,
                    MaxToKeep = max_to_keep,
                    Sharded = sharded,
                    KeepCheckpointEveryNHours = keep_checkpoint_every_n_hours,
                    Version = _write_version
                };
            });
        }

        public Tensor _AddSaveOps(Tensor filename_tensor, SaveableObject[] saveables)
        {
            var save = save_op(filename_tensor, saveables);
            return control_flow_ops.with_dependencies(new Operation[] { save }, filename_tensor);
        }

        /// <summary>
        /// Add operations to restore saveables.
        /// </summary>
        /// <param name="filename_tensor"></param>
        /// <param name="saveables"></param>
        /// <param name="restore_sequentially"></param>
        /// <param name="reshape"></param>
        /// <param name="preferred_shard"></param>
        /// <param name="name"></param>
        /// <returns>An Operation that restores the variables.</returns>
        public Operation _AddRestoreOps(Tensor filename_tensor, 
            SaveableObject[] saveables,
            bool restore_sequentially,
            bool reshape,
            int preferred_shard = -1,
            string name = "restore_all")
        {
            var all_tensors = bulk_restore(filename_tensor, saveables, preferred_shard, restore_sequentially);
            var assign_ops = new List<Tensor>();
            int idx = 0;

            foreach(var saveable in saveables)
            {
                List<TensorShape> shapes = null;
                if (reshape)
                {
                    throw new NotImplementedException("_AddRestoreOps");
                }

                var saveable_tensors = all_tensors.Skip(idx).Take(saveable.specs.Length);
                idx += saveable.specs.Length;
                assign_ops.Add(saveable.restore(saveable_tensors.ToArray(), shapes == null ? null : shapes.ToArray()));
            }

            return control_flow_ops.group(assign_ops.ToArray(), name: name);
        }
    }
}
