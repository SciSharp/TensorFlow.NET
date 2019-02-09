using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Saves and restores variables.
    /// </summary>
    public class Saver
    {
        private RefVariable[] _var_list;
        private bool _reshape;
        private bool _sharded;
        private int _max_to_keep;
        private float _keep_checkpoint_every_n_hours;
        private string _name;
        private bool _restore_sequentially;
        private SaverDef _saver_def;
        private ISaverBuilder _builder;
        private bool _allow_empty;
        private bool _is_built;
        private SaverDef.Types.CheckpointFormatVersion _write_version;
        private bool _pad_step_number;
        private string _filename;
        private bool _is_empty;
        private float _next_checkpoint_time;
        private bool _save_relative_paths;
        private bool? _object_restore_saver;

        public Saver(RefVariable[] var_list = null,
            bool reshape = false,
            bool sharded = false,
            int max_to_keep = 5,
            float keep_checkpoint_every_n_hours = 10000,
            string name = "",
            bool restore_sequentially = false,
            SaverDef saver_def = null,
            ISaverBuilder builder = null,
            bool defer_build = false,
            bool allow_empty = false,
            SaverDef.Types.CheckpointFormatVersion write_version = SaverDef.Types.CheckpointFormatVersion.V2,
            bool pad_step_number = false,
            bool save_relative_paths = false,
            string filename = "")
        {
            _var_list = var_list;
            _reshape = reshape;
            _sharded = sharded;
            _max_to_keep = max_to_keep;
            _keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours;
            _name = name;
            _restore_sequentially = restore_sequentially;
            _builder = builder;
            _is_built = false;
            _allow_empty = allow_empty;
            _write_version = write_version;
            _pad_step_number = pad_step_number;

            if (!defer_build)
                build();
            if(_saver_def != null)
            {
                _check_saver_def();
                _write_version = _saver_def.Version;
            }

            _save_relative_paths = save_relative_paths;
            _object_restore_saver = null;
        }

        public void build()
        {
            _build(_filename, build_save: true, build_restore: true);
        }

        private void _build(string checkpoint_path, bool build_save, bool build_restore)
        {
            if (_is_built) return;

            _is_built = true;

            if (_saver_def == null)
            {
                if (_builder == null)
                    _builder = new BulkSaverBuilder(_write_version);

                if (_var_list == null)
                    _var_list = variables._all_saveable_objects();

                if (_var_list == null || _var_list.Length == 0)
                {
                    if (_allow_empty)
                    {
                        _is_empty = true;
                        return;
                    }
                    else
                    {
                        throw new ValueError("No variables to save");
                    }
                }
                _is_empty = false;

                _saver_def = _builder._build_internal(_var_list,
                    reshape: _reshape,
                    sharded: _sharded,
                    max_to_keep: _max_to_keep,
                    keep_checkpoint_every_n_hours: _keep_checkpoint_every_n_hours,
                    name: _name,
                    restore_sequentially: _restore_sequentially,
                    filename: checkpoint_path,
                    build_save: build_save,
                    build_restore: build_restore);
            }
            else if (_saver_def != null && !string.IsNullOrEmpty(_name))
            {
                throw new NotImplementedException("");
            }

            _check_saver_def();

            _next_checkpoint_time = (float)(DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds + _saver_def.KeepCheckpointEveryNHours * 3600;
        }

        private void _check_saver_def()
        {
            if (!tf.context.executing_eagerly())
            {
                if (string.IsNullOrEmpty(_saver_def.SaveTensorName))
                    throw new ValueError($"saver_def must specify the save_tensor_name: {_saver_def}");
                if (string.IsNullOrEmpty(_saver_def.RestoreOpName))
                    throw new ValueError($"saver_def must specify the restore_op_name: {_saver_def}");
            }
        }

        public string save(Session sess,
            string save_path,
            string global_step = "",
            string meta_graph_suffix = "meta",
            bool write_meta_graph = true,
            bool write_state = true,
            bool strip_default_attrs = false)
        {
            string latest_filename = "checkpoint";
            string model_checkpoint_path = "";
            string checkpoint_file = "";

            if (!string.IsNullOrEmpty(global_step))
            {

            }
            else
            {
                checkpoint_file = save_path;
            }

            var save_path_parent = Path.GetDirectoryName(save_path);

            if (!_is_empty)
            {
                /*model_checkpoint_path = sess.run(_saver_def.SaveTensorName, new FeedItem[] {
                    new FeedItem(_saver_def.FilenameTensorName, checkpoint_file)
                });*/
            }

            throw new NotImplementedException("");

            return model_checkpoint_path;
        }
    }
}
