/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using static Tensorflow.Binding;

namespace Tensorflow;

internal static class gen_io_ops
{
    public static Tensor fixed_length_record_reader(int header_bytes = 0, int record_bytes = 0, int footer_bytes = 0, int hop_bytes = 0, string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FixedLengthRecordReader", name, "header_bytes", header_bytes, "record_bytes", record_bytes, "footer_bytes", footer_bytes, "hop_bytes", hop_bytes, "container", container, "shared_name", shared_name));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return fixed_length_record_reader_eager_fallback(header_bytes: header_bytes, record_bytes: record_bytes, footer_bytes: footer_bytes, hop_bytes: hop_bytes, container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["header_bytes"] = header_bytes; keywords["record_bytes"] = record_bytes; keywords["footer_bytes"] = footer_bytes; keywords["hop_bytes"] = hop_bytes; keywords["container"] = container; keywords["shared_name"] = shared_name; var _op = tf.OpDefLib._apply_op_helper("FixedLengthRecordReader", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "header_bytes", _op._get_attr_int("header_bytes"), "record_bytes", _op._get_attr_int("record_bytes"), "footer_bytes", _op._get_attr_int("footer_bytes"), "hop_bytes", _op._get_attr_int("hop_bytes"), "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("FixedLengthRecordReader", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fixed_length_record_reader_eager_fallback(int header_bytes, int record_bytes, int footer_bytes, int hop_bytes, string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "header_bytes", header_bytes, "record_bytes", record_bytes, "footer_bytes", footer_bytes, "hop_bytes", hop_bytes, "container", container, "shared_name", shared_name };
        var _result = _execute.execute("FixedLengthRecordReader", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FixedLengthRecordReader", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor fixed_length_record_reader_v2(int header_bytes = 0, int record_bytes = 0, int footer_bytes = 0, int hop_bytes = 0, string container = "", string shared_name = "", string encoding = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FixedLengthRecordReaderV2", name, "header_bytes", header_bytes, "record_bytes", record_bytes, "footer_bytes", footer_bytes, "hop_bytes", hop_bytes, "container", container, "shared_name", shared_name, "encoding", encoding));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return fixed_length_record_reader_v2_eager_fallback(header_bytes: header_bytes, record_bytes: record_bytes, footer_bytes: footer_bytes, hop_bytes: hop_bytes, container: container, shared_name: shared_name, encoding: encoding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["header_bytes"] = header_bytes; keywords["record_bytes"] = record_bytes; keywords["footer_bytes"] = footer_bytes; keywords["hop_bytes"] = hop_bytes; keywords["container"] = container; keywords["shared_name"] = shared_name; keywords["encoding"] = encoding; var _op = tf.OpDefLib._apply_op_helper("FixedLengthRecordReaderV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "header_bytes", _op._get_attr_int("header_bytes"), "record_bytes", _op._get_attr_int("record_bytes"), "footer_bytes", _op._get_attr_int("footer_bytes"), "hop_bytes", _op._get_attr_int("hop_bytes"), "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name"), "encoding", _op.get_attr("encoding") };
            _execute.record_gradient("FixedLengthRecordReaderV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fixed_length_record_reader_v2_eager_fallback(int header_bytes, int record_bytes, int footer_bytes, int hop_bytes, string container, string shared_name, string encoding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "header_bytes", header_bytes, "record_bytes", record_bytes, "footer_bytes", footer_bytes, "hop_bytes", hop_bytes, "container", container, "shared_name", shared_name, "encoding", encoding };
        var _result = _execute.execute("FixedLengthRecordReaderV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FixedLengthRecordReaderV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor identity_reader(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IdentityReader", name, "container", container, "shared_name", shared_name));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return identity_reader_eager_fallback(container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container; keywords["shared_name"] = shared_name; var _op = tf.OpDefLib._apply_op_helper("IdentityReader", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("IdentityReader", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor identity_reader_eager_fallback(string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "container", container, "shared_name", shared_name };
        var _result = _execute.execute("IdentityReader", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IdentityReader", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor identity_reader_v2(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IdentityReaderV2", name, "container", container, "shared_name", shared_name));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return identity_reader_v2_eager_fallback(container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container; keywords["shared_name"] = shared_name; var _op = tf.OpDefLib._apply_op_helper("IdentityReaderV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("IdentityReaderV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor identity_reader_v2_eager_fallback(string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "container", container, "shared_name", shared_name };
        var _result = _execute.execute("IdentityReaderV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IdentityReaderV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor matching_files(Tensor pattern, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatchingFiles", name, pattern));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return matching_files_eager_fallback(pattern, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["pattern"] = pattern;
        var _op = tf.OpDefLib._apply_op_helper("MatchingFiles", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("MatchingFiles", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matching_files_eager_fallback(Tensor pattern, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { pattern };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("MatchingFiles", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatchingFiles", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Operation merge_v2_checkpoints(Tensor checkpoint_prefixes, Tensor destination_prefix, bool delete_old_dirs = true, bool allow_missing_files = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MergeV2Checkpoints", name, checkpoint_prefixes, destination_prefix, "delete_old_dirs", delete_old_dirs, "allow_missing_files", allow_missing_files));
                return null;
            }
            catch (Exception)
            {
            }
            try
            {
                return merge_v2_checkpoints_eager_fallback(checkpoint_prefixes, destination_prefix, delete_old_dirs: delete_old_dirs, allow_missing_files: allow_missing_files, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["checkpoint_prefixes"] = checkpoint_prefixes;
        keywords["destination_prefix"] = destination_prefix;
        keywords["delete_old_dirs"] = delete_old_dirs; keywords["allow_missing_files"] = allow_missing_files; var _op = tf.OpDefLib._apply_op_helper("MergeV2Checkpoints", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "delete_old_dirs", _op._get_attr_bool("delete_old_dirs"), "allow_missing_files", _op._get_attr_bool("allow_missing_files") };
            _execute.record_gradient("MergeV2Checkpoints", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor merge_v2_checkpoints_eager_fallback(Tensor checkpoint_prefixes, Tensor destination_prefix, bool delete_old_dirs, bool allow_missing_files, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { checkpoint_prefixes, destination_prefix };
        object[] _attrs = new object[] { "delete_old_dirs", delete_old_dirs, "allow_missing_files", allow_missing_files };
        var _result = _execute.execute("MergeV2Checkpoints", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MergeV2Checkpoints", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    public static Tensor read_file(Tensor filename, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReadFile", name, filename));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return read_file_eager_fallback(filename, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["filename"] = filename;
        var _op = tf.OpDefLib._apply_op_helper("ReadFile", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReadFile", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor read_file_eager_fallback(Tensor filename, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { filename };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReadFile", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReadFile", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor reader_num_records_produced(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("reader_num_records_produced op does not support eager execution. Arg reader_handle is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderNumRecordsProduced", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderNumRecordsProduced", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reader_num_records_produced_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        throw new RuntimeError($"reader_num_records_produced op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    public static Tensor reader_num_records_produced_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderNumRecordsProducedV2", name, reader_handle));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return reader_num_records_produced_v2_eager_fallback(reader_handle, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderNumRecordsProducedV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderNumRecordsProducedV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reader_num_records_produced_v2_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { reader_handle };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReaderNumRecordsProducedV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReaderNumRecordsProducedV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor reader_num_work_units_completed(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("reader_num_work_units_completed op does not support eager execution. Arg reader_handle is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderNumWorkUnitsCompleted", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderNumWorkUnitsCompleted", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reader_num_work_units_completed_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        throw new RuntimeError($"reader_num_work_units_completed op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    public static Tensor reader_num_work_units_completed_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderNumWorkUnitsCompletedV2", name, reader_handle));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return reader_num_work_units_completed_v2_eager_fallback(reader_handle, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderNumWorkUnitsCompletedV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderNumWorkUnitsCompletedV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reader_num_work_units_completed_v2_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { reader_handle };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReaderNumWorkUnitsCompletedV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReaderNumWorkUnitsCompletedV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor[] reader_read(Tensor reader_handle, Tensor queue_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("reader_read op does not support eager execution. Arg reader_handle is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        keywords["queue_handle"] = queue_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderRead", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderRead", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] reader_read_eager_fallback(Tensor reader_handle, Tensor queue_handle, string name, Context ctx)
    {
        throw new RuntimeError($"reader_read op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    public static Tensor[] reader_read_up_to(Tensor reader_handle, Tensor queue_handle, Tensor num_records, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("reader_read_up_to op does not support eager execution. Arg reader_handle is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        keywords["queue_handle"] = queue_handle;
        keywords["num_records"] = num_records;
        var _op = tf.OpDefLib._apply_op_helper("ReaderReadUpTo", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderReadUpTo", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] reader_read_up_to_eager_fallback(Tensor reader_handle, Tensor queue_handle, Tensor num_records, string name, Context ctx)
    {
        throw new RuntimeError($"reader_read_up_to op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    public static Tensor[] reader_read_up_to_v2(Tensor reader_handle, Tensor queue_handle, Tensor num_records, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderReadUpToV2", name, reader_handle, queue_handle, num_records));
                return _fast_path_result;
            }
            catch (Exception)
            {
            }
            try
            {
                return reader_read_up_to_v2_eager_fallback(reader_handle, queue_handle, num_records, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        keywords["queue_handle"] = queue_handle;
        keywords["num_records"] = num_records;
        var _op = tf.OpDefLib._apply_op_helper("ReaderReadUpToV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderReadUpToV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] reader_read_up_to_v2_eager_fallback(Tensor reader_handle, Tensor queue_handle, Tensor num_records, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { reader_handle, queue_handle, num_records };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReaderReadUpToV2", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReaderReadUpToV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    public static Tensor[] reader_read_v2(Tensor reader_handle, Tensor queue_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderReadV2", name, reader_handle, queue_handle));
                return _fast_path_result;
            }
            catch (Exception)
            {
            }
            try
            {
                return reader_read_v2_eager_fallback(reader_handle, queue_handle, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        keywords["queue_handle"] = queue_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderReadV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderReadV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] reader_read_v2_eager_fallback(Tensor reader_handle, Tensor queue_handle, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { reader_handle, queue_handle };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReaderReadV2", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReaderReadV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    public static Operation reader_reset(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("reader_reset op does not support eager execution. Arg reader_handle is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderReset", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderReset", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor reader_reset_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        throw new RuntimeError($"reader_reset op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    public static Operation reader_reset_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderResetV2", name, reader_handle));
                return null;
            }
            catch (Exception)
            {
            }
            try
            {
                return reader_reset_v2_eager_fallback(reader_handle, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderResetV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderResetV2", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor reader_reset_v2_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { reader_handle };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReaderResetV2", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReaderResetV2", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    public static Operation reader_restore_state(Tensor reader_handle, Tensor state, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("reader_restore_state op does not support eager execution. Arg reader_handle is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        keywords["state"] = state;
        var _op = tf.OpDefLib._apply_op_helper("ReaderRestoreState", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderRestoreState", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor reader_restore_state_eager_fallback(Tensor reader_handle, Tensor state, string name, Context ctx)
    {
        throw new RuntimeError($"reader_restore_state op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    public static Operation reader_restore_state_v2(Tensor reader_handle, Tensor state, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderRestoreStateV2", name, reader_handle, state));
                return null;
            }
            catch (Exception)
            {
            }
            try
            {
                return reader_restore_state_v2_eager_fallback(reader_handle, state, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        keywords["state"] = state;
        var _op = tf.OpDefLib._apply_op_helper("ReaderRestoreStateV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderRestoreStateV2", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor reader_restore_state_v2_eager_fallback(Tensor reader_handle, Tensor state, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { reader_handle, state };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReaderRestoreStateV2", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReaderRestoreStateV2", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    public static Tensor reader_serialize_state(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("reader_serialize_state op does not support eager execution. Arg reader_handle is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderSerializeState", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderSerializeState", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reader_serialize_state_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        throw new RuntimeError($"reader_serialize_state op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    public static Tensor reader_serialize_state_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderSerializeStateV2", name, reader_handle));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return reader_serialize_state_v2_eager_fallback(reader_handle, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["reader_handle"] = reader_handle;
        var _op = tf.OpDefLib._apply_op_helper("ReaderSerializeStateV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ReaderSerializeStateV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reader_serialize_state_v2_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { reader_handle };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ReaderSerializeStateV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReaderSerializeStateV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor restore(Tensor file_pattern, Tensor tensor_name, TF_DataType dt, int preferred_shard = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Restore", name, file_pattern, tensor_name, "dt", dt, "preferred_shard", preferred_shard));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return restore_eager_fallback(file_pattern, tensor_name, dt: dt, preferred_shard: preferred_shard, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["file_pattern"] = file_pattern;
        keywords["tensor_name"] = tensor_name;
        keywords["dt"] = dt; keywords["preferred_shard"] = preferred_shard; var _op = tf.OpDefLib._apply_op_helper("Restore", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dt", _op._get_attr_type("dt"), "preferred_shard", _op._get_attr_int("preferred_shard") };
            _execute.record_gradient("Restore", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor restore_eager_fallback(Tensor file_pattern, Tensor tensor_name, TF_DataType dt, int preferred_shard, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { file_pattern, tensor_name };
        object[] _attrs = new object[] { "dt", dt, "preferred_shard", preferred_shard };
        var _result = _execute.execute("Restore", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Restore", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor restore_slice(Tensor file_pattern, Tensor tensor_name, Tensor shape_and_slice, TF_DataType dt, int preferred_shard = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RestoreSlice", name, file_pattern, tensor_name, shape_and_slice, "dt", dt, "preferred_shard", preferred_shard));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return restore_slice_eager_fallback(file_pattern, tensor_name, shape_and_slice, dt: dt, preferred_shard: preferred_shard, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["file_pattern"] = file_pattern;
        keywords["tensor_name"] = tensor_name;
        keywords["shape_and_slice"] = shape_and_slice;
        keywords["dt"] = dt; keywords["preferred_shard"] = preferred_shard; var _op = tf.OpDefLib._apply_op_helper("RestoreSlice", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dt", _op._get_attr_type("dt"), "preferred_shard", _op._get_attr_int("preferred_shard") };
            _execute.record_gradient("RestoreSlice", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor restore_slice_eager_fallback(Tensor file_pattern, Tensor tensor_name, Tensor shape_and_slice, TF_DataType dt, int preferred_shard, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { file_pattern, tensor_name, shape_and_slice };
        object[] _attrs = new object[] { "dt", dt, "preferred_shard", preferred_shard };
        var _result = _execute.execute("RestoreSlice", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RestoreSlice", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor restore_v2(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, TF_DataType[] dtypes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RestoreV2", name, prefix, tensor_names, shape_and_slices, "dtypes", dtypes));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return restore_v2_eager_fallback(prefix, tensor_names, shape_and_slices, dtypes: dtypes, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["prefix"] = prefix;
        keywords["tensor_names"] = tensor_names;
        keywords["shape_and_slices"] = shape_and_slices;
        keywords["dtypes"] = dtypes; var _op = tf.OpDefLib._apply_op_helper("RestoreV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtypes", _op.get_attr("dtypes") };
            _execute.record_gradient("RestoreV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor restore_v2_eager_fallback(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, TF_DataType[] dtypes, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { prefix, tensor_names, shape_and_slices };
        object[] _attrs = new object[] { "dtypes", dtypes };
        var _result = _execute.execute("RestoreV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RestoreV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Operation save(Tensor filename, Tensor tensor_names, Tensor data, TF_DataType[] T, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Save", name, filename, tensor_names, data, "T", T));
                return null;
            }
            catch (Exception)
            {
            }
            try
            {
                return save_eager_fallback(filename, tensor_names, data, T: T, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["filename"] = filename;
        keywords["tensor_names"] = tensor_names;
        keywords["data"] = data;
        keywords["T"] = T; var _op = tf.OpDefLib._apply_op_helper("Save", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T") };
            _execute.record_gradient("Save", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor save_eager_fallback(Tensor filename, Tensor tensor_names, Tensor data, TF_DataType[] T, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { filename, tensor_names, data };
        object[] _attrs = new object[] { "T", T };
        var _result = _execute.execute("Save", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Save", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    public static Operation save_slices(Tensor filename, Tensor tensor_names, Tensor shapes_and_slices, Tensor data, TF_DataType[] T, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SaveSlices", name, filename, tensor_names, shapes_and_slices, data, "T", T));
                return null;
            }
            catch (Exception)
            {
            }
            try
            {
                return save_slices_eager_fallback(filename, tensor_names, shapes_and_slices, data, T: T, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["filename"] = filename;
        keywords["tensor_names"] = tensor_names;
        keywords["shapes_and_slices"] = shapes_and_slices;
        keywords["data"] = data;
        keywords["T"] = T; var _op = tf.OpDefLib._apply_op_helper("SaveSlices", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T") };
            _execute.record_gradient("SaveSlices", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor save_slices_eager_fallback(Tensor filename, Tensor tensor_names, Tensor shapes_and_slices, Tensor data, TF_DataType[] T, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { filename, tensor_names, shapes_and_slices, data };
        object[] _attrs = new object[] { "T", T };
        var _result = _execute.execute("SaveSlices", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SaveSlices", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    public static Operation save_v2(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, Tensor tensors, TF_DataType[] dtypes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SaveV2", name, prefix, tensor_names, shape_and_slices, tensors, "dtypes", dtypes));
                return null;
            }
            catch (Exception)
            {
            }
            try
            {
                return save_v2_eager_fallback(prefix, tensor_names, shape_and_slices, tensors, dtypes: dtypes, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["prefix"] = prefix;
        keywords["tensor_names"] = tensor_names;
        keywords["shape_and_slices"] = shape_and_slices;
        keywords["tensors"] = tensors;
        keywords["dtypes"] = dtypes; var _op = tf.OpDefLib._apply_op_helper("SaveV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtypes", _op.get_attr("dtypes") };
            _execute.record_gradient("SaveV2", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor save_v2_eager_fallback(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, Tensor tensors, TF_DataType[] dtypes, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { prefix, tensor_names, shape_and_slices, tensors };
        object[] _attrs = new object[] { "dtypes", dtypes };
        var _result = _execute.execute("SaveV2", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SaveV2", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    public static Tensor sharded_filename(Tensor basename, Tensor shard, Tensor num_shards, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ShardedFilename", name, basename, shard, num_shards));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return sharded_filename_eager_fallback(basename, shard, num_shards, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["basename"] = basename;
        keywords["shard"] = shard;
        keywords["num_shards"] = num_shards;
        var _op = tf.OpDefLib._apply_op_helper("ShardedFilename", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ShardedFilename", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sharded_filename_eager_fallback(Tensor basename, Tensor shard, Tensor num_shards, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { basename, shard, num_shards };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ShardedFilename", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ShardedFilename", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor sharded_filespec(Tensor basename, Tensor num_shards, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ShardedFilespec", name, basename, num_shards));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return sharded_filespec_eager_fallback(basename, num_shards, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["basename"] = basename;
        keywords["num_shards"] = num_shards;
        var _op = tf.OpDefLib._apply_op_helper("ShardedFilespec", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("ShardedFilespec", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sharded_filespec_eager_fallback(Tensor basename, Tensor num_shards, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { basename, num_shards };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("ShardedFilespec", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ShardedFilespec", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor text_line_reader(int skip_header_lines = 0, string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TextLineReader", name, "skip_header_lines", skip_header_lines, "container", container, "shared_name", shared_name));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return text_line_reader_eager_fallback(skip_header_lines: skip_header_lines, container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["skip_header_lines"] = skip_header_lines; keywords["container"] = container; keywords["shared_name"] = shared_name; var _op = tf.OpDefLib._apply_op_helper("TextLineReader", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "skip_header_lines", _op._get_attr_int("skip_header_lines"), "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("TextLineReader", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor text_line_reader_eager_fallback(int skip_header_lines, string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "skip_header_lines", skip_header_lines, "container", container, "shared_name", shared_name };
        var _result = _execute.execute("TextLineReader", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TextLineReader", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor text_line_reader_v2(int skip_header_lines = 0, string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TextLineReaderV2", name, "skip_header_lines", skip_header_lines, "container", container, "shared_name", shared_name));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return text_line_reader_v2_eager_fallback(skip_header_lines: skip_header_lines, container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["skip_header_lines"] = skip_header_lines; keywords["container"] = container; keywords["shared_name"] = shared_name; var _op = tf.OpDefLib._apply_op_helper("TextLineReaderV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "skip_header_lines", _op._get_attr_int("skip_header_lines"), "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("TextLineReaderV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor text_line_reader_v2_eager_fallback(int skip_header_lines, string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "skip_header_lines", skip_header_lines, "container", container, "shared_name", shared_name };
        var _result = _execute.execute("TextLineReaderV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TextLineReaderV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor whole_file_reader(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "WholeFileReader", name, "container", container, "shared_name", shared_name));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return whole_file_reader_eager_fallback(container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container; keywords["shared_name"] = shared_name; var _op = tf.OpDefLib._apply_op_helper("WholeFileReader", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("WholeFileReader", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor whole_file_reader_eager_fallback(string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "container", container, "shared_name", shared_name };
        var _result = _execute.execute("WholeFileReader", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("WholeFileReader", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Tensor whole_file_reader_v2(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "WholeFileReaderV2", name, "container", container, "shared_name", shared_name));
                return _fast_path_result[0];
            }
            catch (Exception)
            {
            }
            try
            {
                return whole_file_reader_v2_eager_fallback(container: container, shared_name: shared_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container; keywords["shared_name"] = shared_name; var _op = tf.OpDefLib._apply_op_helper("WholeFileReaderV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "container", _op.get_attr("container"), "shared_name", _op.get_attr("shared_name") };
            _execute.record_gradient("WholeFileReaderV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor whole_file_reader_v2_eager_fallback(string container, string shared_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "container", container, "shared_name", shared_name };
        var _result = _execute.execute("WholeFileReaderV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("WholeFileReaderV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    public static Operation write_file(Tensor filename, Tensor contents, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "WriteFile", name, filename, contents));
                return null;
            }
            catch (Exception)
            {
            }
            try
            {
                return write_file_eager_fallback(filename, contents, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["filename"] = filename;
        keywords["contents"] = contents;
        var _op = tf.OpDefLib._apply_op_helper("WriteFile", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("WriteFile", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Tensor write_file_eager_fallback(Tensor filename, Tensor contents, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { filename, contents };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("WriteFile", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("WriteFile", _inputs_flat, _attrs, _result);
        }
        return null;
    }
}
