/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow;

public static class gen_io_ops
{
    /// <summary>
    /// A Reader that outputs fixed-length records from a file.
    /// </summary>
    /// <param name="header_bytes">
    /// 
    /// Number of bytes in the header, defaults to 0.
    /// 
    /// </param>
    /// <param name="record_bytes">
    /// 
    /// Number of bytes in the record.
    /// 
    /// </param>
    /// <param name="footer_bytes">
    /// 
    /// Number of bytes in the footer, defaults to 0.
    /// 
    /// </param>
    /// <param name="hop_bytes">
    /// 
    /// Number of bytes to hop before each read. Default of 0 means using
    /// record_bytes.
    /// 
    /// </param>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor fixed_length_record_reader(int header_bytes = 0, int record_bytes = 0, int footer_bytes = 0, int hop_bytes = 0, string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FixedLengthRecordReader", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["header_bytes"] = header_bytes, ["record_bytes"] = record_bytes, ["footer_bytes"] = footer_bytes, ["hop_bytes"] = hop_bytes, ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["header_bytes"] = header_bytes;
        keywords["record_bytes"] = record_bytes;
        keywords["footer_bytes"] = footer_bytes;
        keywords["hop_bytes"] = hop_bytes;
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("FixedLengthRecordReader", name, keywords);
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
    /// <summary>
    /// A Reader that outputs fixed-length records from a file.
    /// </summary>
    /// <param name="header_bytes">
    /// 
    /// Number of bytes in the header, defaults to 0.
    /// 
    /// </param>
    /// <param name="record_bytes">
    /// 
    /// Number of bytes in the record.
    /// 
    /// </param>
    /// <param name="footer_bytes">
    /// 
    /// Number of bytes in the footer, defaults to 0.
    /// 
    /// </param>
    /// <param name="hop_bytes">
    /// 
    /// Number of bytes to hop before each read. Default of 0 means using
    /// record_bytes.
    /// 
    /// </param>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <param name="encoding">
    /// 
    /// The type of encoding for the file. Currently ZLIB and GZIP
    /// are supported. Defaults to none.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor fixed_length_record_reader_v2(int header_bytes = 0, int record_bytes = 0, int footer_bytes = 0, int hop_bytes = 0, string container = "", string shared_name = "", string encoding = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FixedLengthRecordReaderV2", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["header_bytes"] = header_bytes, ["record_bytes"] = record_bytes, ["footer_bytes"] = footer_bytes, ["hop_bytes"] = hop_bytes, ["container"] = container, ["shared_name"] = shared_name, ["encoding"] = encoding } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        if (encoding is null)
        {
            encoding = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["header_bytes"] = header_bytes;
        keywords["record_bytes"] = record_bytes;
        keywords["footer_bytes"] = footer_bytes;
        keywords["hop_bytes"] = hop_bytes;
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        keywords["encoding"] = encoding;
        var _op = tf.OpDefLib._apply_op_helper("FixedLengthRecordReaderV2", name, keywords);
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
    /// <summary>
    /// A Reader that outputs the queued work as both the key and value.
    /// </summary>
    /// <remarks>
    /// 
    /// To use, enqueue strings in a Queue.  ReaderRead will take the front
    /// work string and output (work, work).
    /// 
    /// </remarks>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor identity_reader(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IdentityReader", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("IdentityReader", name, keywords);
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
    /// <summary>
    /// A Reader that outputs the queued work as both the key and value.
    /// </summary>
    /// <remarks>
    /// 
    /// To use, enqueue strings in a Queue.  ReaderRead will take the front
    /// work string and output (work, work).
    /// 
    /// </remarks>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor identity_reader_v2(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IdentityReaderV2", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("IdentityReaderV2", name, keywords);
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
    /// <summary>
    /// Returns the set of files matching one or more glob patterns.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that this routine only supports wildcard characters in the
    /// basename portion of the pattern, not in the directory portion.
    /// Note also that the order of filenames returned is deterministic.
    /// 
    /// </remarks>
    /// <param name="pattern"></param>
    /// <returns></returns>
    public static Tensor matching_files(Tensor pattern, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatchingFiles", name) { args = new object[] { pattern }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Reads and outputs the entire contents of the input filename.
    /// </summary>
    /// <param name="filename"></param>
    /// <returns></returns>
    public static Tensor read_file(Tensor filename, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReadFile", name) { args = new object[] { filename }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Returns the number of records this Reader has produced.
    /// </summary>
    /// <remarks>
    /// 
    /// This is the same as the number of ReaderRead executions that have
    /// succeeded.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
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
    /// <summary>
    /// Returns the number of records this Reader has produced.
    /// </summary>
    /// <remarks>
    /// 
    /// This is the same as the number of ReaderRead executions that have
    /// succeeded.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
    public static Tensor reader_num_records_produced_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderNumRecordsProducedV2", name) { args = new object[] { reader_handle }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Returns the number of work units this Reader has finished processing.
    /// </summary>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
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
    /// <summary>
    /// Returns the number of work units this Reader has finished processing.
    /// </summary>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
    public static Tensor reader_num_work_units_completed_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderNumWorkUnitsCompletedV2", name) { args = new object[] { reader_handle }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Returns the next record (key, value pair) produced by a Reader.
    /// </summary>
    /// <remarks>
    /// 
    /// Will dequeue from the input queue if necessary (e.g. when the
    /// Reader needs to start reading from a new file since it has finished
    /// with the previous file).
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <param name="queue_handle"></param>
    /// <returns></returns>
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
    /// <summary>
    /// Returns up to `num_records` (key, value) pairs produced by a Reader.
    /// </summary>
    /// <remarks>
    /// 
    /// Will dequeue from the input queue if necessary (e.g. when the
    /// Reader needs to start reading from a new file since it has finished
    /// with the previous file).
    /// It may return less than `num_records` even before the last batch.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <param name="queue_handle"></param>
    /// <param name="num_records"></param>
    /// <returns></returns>
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
    /// <summary>
    /// Returns up to `num_records` (key, value) pairs produced by a Reader.
    /// </summary>
    /// <remarks>
    /// 
    /// Will dequeue from the input queue if necessary (e.g. when the
    /// Reader needs to start reading from a new file since it has finished
    /// with the previous file).
    /// It may return less than `num_records` even before the last batch.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <param name="queue_handle"></param>
    /// <param name="num_records"></param>
    /// <returns></returns>
    public static Tensor[] reader_read_up_to_v2(Tensor reader_handle, Tensor queue_handle, Tensor num_records, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderReadUpToV2", name) { args = new object[] { reader_handle, queue_handle, num_records }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Returns the next record (key, value pair) produced by a Reader.
    /// </summary>
    /// <remarks>
    /// 
    /// Will dequeue from the input queue if necessary (e.g. when the
    /// Reader needs to start reading from a new file since it has finished
    /// with the previous file).
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <param name="queue_handle"></param>
    /// <returns></returns>
    public static Tensor[] reader_read_v2(Tensor reader_handle, Tensor queue_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderReadV2", name) { args = new object[] { reader_handle, queue_handle }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Restore a Reader to its initial clean state.
    /// </summary>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
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

    public static Operation reader_reset_eager_fallback(Tensor reader_handle, string name, Context ctx)
    {
        throw new RuntimeError($"reader_reset op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    /// <summary>
    /// Restore a Reader to its initial clean state.
    /// </summary>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
    public static Operation reader_reset_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderResetV2", name) { args = new object[] { reader_handle }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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

    public static Operation reader_reset_v2_eager_fallback(Tensor reader_handle, string name, Context ctx)
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
    /// <summary>
    /// Restore a reader to a previously saved state.
    /// </summary>
    /// <remarks>
    /// 
    /// Not all Readers support being restored, so this can produce an
    /// Unimplemented error.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <param name="state"></param>
    /// <returns></returns>
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

    public static Operation reader_restore_state_eager_fallback(Tensor reader_handle, Tensor state, string name, Context ctx)
    {
        throw new RuntimeError($"reader_restore_state op does not support eager execution. Arg 'reader_handle' is a ref.");
    }
    /// <summary>
    /// Restore a reader to a previously saved state.
    /// </summary>
    /// <remarks>
    /// 
    /// Not all Readers support being restored, so this can produce an
    /// Unimplemented error.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <param name="state"></param>
    /// <returns></returns>
    public static Operation reader_restore_state_v2(Tensor reader_handle, Tensor state, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderRestoreStateV2", name) { args = new object[] { reader_handle, state }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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

    public static Operation reader_restore_state_v2_eager_fallback(Tensor reader_handle, Tensor state, string name, Context ctx)
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
    /// <summary>
    /// Produce a string tensor that encodes the state of a Reader.
    /// </summary>
    /// <remarks>
    /// 
    /// Not all Readers support being serialized, so this can produce an
    /// Unimplemented error.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
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
    /// <summary>
    /// Produce a string tensor that encodes the state of a Reader.
    /// </summary>
    /// <remarks>
    /// 
    /// Not all Readers support being serialized, so this can produce an
    /// Unimplemented error.
    /// 
    /// </remarks>
    /// <param name="reader_handle"></param>
    /// <returns></returns>
    public static Tensor reader_serialize_state_v2(Tensor reader_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReaderSerializeStateV2", name) { args = new object[] { reader_handle }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Restores a tensor from checkpoint files.
    /// </summary>
    /// <remarks>
    /// 
    /// Reads a tensor stored in one or several files. If there are several files (for
    /// instance because a tensor was saved as slices), `file_pattern` may contain
    /// wildcard symbols (`*` and `?`) in the filename portion only, not in the
    /// directory portion.
    /// 
    /// If a `file_pattern` matches several files, `preferred_shard` can be used to hint
    /// in which file the requested tensor is likely to be found. This op will first
    /// open the file at index `preferred_shard` in the list of matching files and try
    /// to restore tensors from that file.  Only if some tensors or tensor slices are
    /// not found in that first file, then the Op opens all the files. Setting
    /// `preferred_shard` to match the value passed as the `shard` input
    /// of a matching `Save` Op may speed up Restore.  This attribute only affects
    /// performance, not correctness.  The default value -1 means files are processed in
    /// order.
    /// 
    /// See also `RestoreSlice`.
    /// 
    /// </remarks>
    /// <param name="file_pattern"></param>
    /// <param name="tensor_name"></param>
    /// <param name="dt">
    /// 
    /// The type of the tensor to be restored.
    /// 
    /// </param>
    /// <param name="preferred_shard">
    /// 
    /// Index of file to open first if multiple files match
    /// `file_pattern`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor restore(Tensor file_pattern, Tensor tensor_name, TF_DataType dt, int preferred_shard = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Restore", name) { args = new object[] { file_pattern, tensor_name }, attrs = new Dictionary<string, object>() { ["dt"] = dt, ["preferred_shard"] = preferred_shard } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        keywords["dt"] = dt;
        keywords["preferred_shard"] = preferred_shard;
        var _op = tf.OpDefLib._apply_op_helper("Restore", name, keywords);
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
    /// <summary>
    /// Restores a tensor from checkpoint files.
    /// </summary>
    /// <remarks>
    /// 
    /// This is like `Restore` except that restored tensor can be listed as filling
    /// only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
    /// larger tensor and the slice that the restored tensor covers.
    /// 
    /// The `shape_and_slice` input has the same format as the
    /// elements of the `shapes_and_slices` input of the `SaveSlices` op.
    /// 
    /// </remarks>
    /// <param name="file_pattern"></param>
    /// <param name="tensor_name"></param>
    /// <param name="shape_and_slice"></param>
    /// <param name="dt">
    /// 
    /// The type of the tensor to be restored.
    /// 
    /// </param>
    /// <param name="preferred_shard">
    /// 
    /// Index of file to open first if multiple files match
    /// `file_pattern`. See the documentation for `Restore`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor restore_slice(Tensor file_pattern, Tensor tensor_name, Tensor shape_and_slice, TF_DataType dt, int preferred_shard = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RestoreSlice", name) { args = new object[] { file_pattern, tensor_name, shape_and_slice }, attrs = new Dictionary<string, object>() { ["dt"] = dt, ["preferred_shard"] = preferred_shard } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        keywords["dt"] = dt;
        keywords["preferred_shard"] = preferred_shard;
        var _op = tf.OpDefLib._apply_op_helper("RestoreSlice", name, keywords);
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
    /// <summary>
    /// Restores tensors from a V2 checkpoint.
    /// </summary>
    /// <remarks>
    /// 
    /// For backward compatibility with the V1 format, this Op currently allows
    /// restoring from a V1 checkpoint as well:
    ///   - This Op first attempts to find the V2 index file pointed to by "prefix", and
    ///     if found proceed to read it as a V2 checkpoint;
    ///   - Otherwise the V1 read path is invoked.
    /// Relying on this behavior is not recommended, as the ability to fall back to read
    /// V1 might be deprecated and eventually removed.
    /// 
    /// By default, restores the named tensors in full.  If the caller wishes to restore
    /// specific slices of stored tensors, "shape_and_slices" should be non-empty
    /// strings and correspondingly well-formed.
    /// 
    /// Callers must ensure all the named tensors are indeed stored in the checkpoint.
    /// 
    /// </remarks>
    /// <param name="prefix"></param>
    /// <param name="tensor_names"></param>
    /// <param name="shape_and_slices"></param>
    /// <param name="dtypes">
    /// 
    /// shape {N}.  The list of expected dtype for the tensors.  Must match
    /// those stored in the checkpoint.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] restore_v2(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, TF_DataType[] dtypes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RestoreV2", name) { args = new object[] { prefix, tensor_names, shape_and_slices }, attrs = new Dictionary<string, object>() { ["dtypes"] = dtypes } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        keywords["dtypes"] = dtypes;
        var _op = tf.OpDefLib._apply_op_helper("RestoreV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtypes", _op.get_attr("dtypes") };
            _execute.record_gradient("RestoreV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] restore_v2_eager_fallback(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, TF_DataType[] dtypes, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { prefix, tensor_names, shape_and_slices };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("RestoreV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RestoreV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Saves the input tensors to disk.
    /// </summary>
    /// <remarks>
    /// 
    /// The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
    /// is written to `filename` with name `tensor_names[i]`.
    /// 
    /// See also `SaveSlices`.
    /// 
    /// </remarks>
    /// <param name="filename"></param>
    /// <param name="tensor_names"></param>
    /// <param name="data"></param>
    /// <returns></returns>
    public static Operation save(Tensor filename, Tensor tensor_names, Tensors data, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Save", name) { args = new object[] { filename, tensor_names, data }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return save_eager_fallback(filename, tensor_names, data, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["filename"] = filename;
        keywords["tensor_names"] = tensor_names;
        keywords["data"] = data;
        var _op = tf.OpDefLib._apply_op_helper("Save", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T") };
            _execute.record_gradient("Save", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation save_eager_fallback(Tensor filename, Tensor tensor_names, Tensor data, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { filename, tensor_names, data };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("Save", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Save", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Saves input tensors slices to disk.
    /// </summary>
    /// <remarks>
    /// 
    /// This is like `Save` except that tensors can be listed in the saved file as being
    /// a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
    /// larger tensor and the slice that this tensor covers. `shapes_and_slices` must
    /// have as many elements as `tensor_names`.
    /// 
    /// Elements of the `shapes_and_slices` input must either be:
    /// 
    /// *  The empty string, in which case the corresponding tensor is
    ///    saved normally.
    /// *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
    ///    `dimI` are the dimensions of the larger tensor and `slice-spec`
    ///    specifies what part is covered by the tensor to save.
    /// 
    /// `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
    /// where each `sliceI` is either:
    /// 
    /// *  The string `-` meaning that the slice covers all indices of this dimension
    /// *  `start,length` where `start` and `length` are integers.  In that
    ///    case the slice covers `length` indices starting at `start`.
    /// 
    /// See also `Save`.
    /// 
    /// </remarks>
    /// <param name="filename"></param>
    /// <param name="tensor_names"></param>
    /// <param name="shapes_and_slices"></param>
    /// <param name="data"></param>
    /// <returns></returns>
    public static Operation save_slices(Tensor filename, Tensor tensor_names, Tensor shapes_and_slices, Tensors data, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SaveSlices", name) { args = new object[] { filename, tensor_names, shapes_and_slices, data }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return save_slices_eager_fallback(filename, tensor_names, shapes_and_slices, data, name: name, ctx: _ctx);
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
        var _op = tf.OpDefLib._apply_op_helper("SaveSlices", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T") };
            _execute.record_gradient("SaveSlices", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation save_slices_eager_fallback(Tensor filename, Tensor tensor_names, Tensor shapes_and_slices, Tensor data, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { filename, tensor_names, shapes_and_slices, data };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("SaveSlices", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SaveSlices", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Saves tensors in V2 checkpoint format.
    /// </summary>
    /// <remarks>
    /// 
    /// By default, saves the named tensors in full.  If the caller wishes to save
    /// specific slices of full tensors, "shape_and_slices" should be non-empty strings
    /// and correspondingly well-formed.
    /// 
    /// </remarks>
    /// <param name="prefix"></param>
    /// <param name="tensor_names"></param>
    /// <param name="shape_and_slices"></param>
    /// <param name="tensors"></param>
    /// <returns></returns>
    public static Operation save_v2(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, Tensors tensors, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SaveV2", name) { args = new object[] { prefix, tensor_names, shape_and_slices, tensors }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return save_v2_eager_fallback(prefix, tensor_names, shape_and_slices, tensors, name: name, ctx: _ctx);
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
        var _op = tf.OpDefLib._apply_op_helper("SaveV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtypes", _op.get_attr("dtypes") };
            _execute.record_gradient("SaveV2", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation save_v2_eager_fallback(Tensor prefix, Tensor tensor_names, Tensor shape_and_slices, Tensor tensors, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { prefix, tensor_names, shape_and_slices, tensors };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("SaveV2", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SaveV2", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Generate a sharded filename. The filename is printf formatted as
    /// </summary>
    /// <remarks>
    /// 
    ///    %s-%05d-of-%05d, basename, shard, num_shards.
    /// 
    /// </remarks>
    /// <param name="basename"></param>
    /// <param name="shard"></param>
    /// <param name="num_shards"></param>
    /// <returns></returns>
    public static Tensor sharded_filename(Tensor basename, Tensor shard, Tensor num_shards, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ShardedFilename", name) { args = new object[] { basename, shard, num_shards }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// Generate a glob pattern matching all sharded file names.
    /// </summary>
    /// <param name="basename"></param>
    /// <param name="num_shards"></param>
    /// <returns></returns>
    public static Tensor sharded_filespec(Tensor basename, Tensor num_shards, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ShardedFilespec", name) { args = new object[] { basename, num_shards }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
    /// <summary>
    /// A Reader that outputs the lines of a file delimited by '\n'.
    /// </summary>
    /// <param name="skip_header_lines">
    /// 
    /// Number of lines to skip from the beginning of every file.
    /// 
    /// </param>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor text_line_reader(int skip_header_lines = 0, string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TextLineReader", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["skip_header_lines"] = skip_header_lines, ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["skip_header_lines"] = skip_header_lines;
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("TextLineReader", name, keywords);
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
    /// <summary>
    /// A Reader that outputs the lines of a file delimited by '\n'.
    /// </summary>
    /// <param name="skip_header_lines">
    /// 
    /// Number of lines to skip from the beginning of every file.
    /// 
    /// </param>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor text_line_reader_v2(int skip_header_lines = 0, string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TextLineReaderV2", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["skip_header_lines"] = skip_header_lines, ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["skip_header_lines"] = skip_header_lines;
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("TextLineReaderV2", name, keywords);
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
    /// <summary>
    /// A Reader that outputs the entire contents of a file as a value.
    /// </summary>
    /// <remarks>
    /// 
    /// To use, enqueue filenames in a Queue.  The output of ReaderRead will
    /// be a filename (key) and the contents of that file (value).
    /// 
    /// </remarks>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor whole_file_reader(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "WholeFileReader", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("WholeFileReader", name, keywords);
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
    /// <summary>
    /// A Reader that outputs the entire contents of a file as a value.
    /// </summary>
    /// <remarks>
    /// 
    /// To use, enqueue filenames in a Queue.  The output of ReaderRead will
    /// be a filename (key) and the contents of that file (value).
    /// 
    /// </remarks>
    /// <param name="container">
    /// 
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    /// 
    /// </param>
    /// <param name="shared_name">
    /// 
    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor whole_file_reader_v2(string container = "", string shared_name = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "WholeFileReaderV2", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["container"] = container, ["shared_name"] = shared_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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
        if (container is null)
        {
            container = "";
        }
        if (shared_name is null)
        {
            shared_name = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["container"] = container;
        keywords["shared_name"] = shared_name;
        var _op = tf.OpDefLib._apply_op_helper("WholeFileReaderV2", name, keywords);
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
    /// <summary>
    /// Writes `contents` to the file at input `filename`.
    /// </summary>
    /// <remarks>
    /// 
    /// Creates the file and recursively creates directory if it does not exist.
    /// 
    /// </remarks>
    /// <param name="filename"></param>
    /// <param name="contents"></param>
    /// <returns></returns>
    public static Operation write_file(Tensor filename, Tensor contents, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "WriteFile", name) { args = new object[] { filename, contents }, attrs = new Dictionary<string, object>() { } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
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

    public static Operation write_file_eager_fallback(Tensor filename, Tensor contents, string name, Context ctx)
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
