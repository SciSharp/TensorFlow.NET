using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Train;
using static Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types;
using static Tensorflow.Binding;

namespace Tensorflow.Checkpoint;

/// <summary>
/// Saves and restores a `Trackable` object and its dependencies.
/// </summary>
public class TrackableSaver
{
    private ObjectGraphView _graph_view;
    private Tensor _cached_save_operation;
    private TrackableObjectGraph _last_save_object_graph;
    private Tensor? _object_graph_feed_tensor = null;
    private Tensor? _file_prefix_feed_tensor = null;
    private Dictionary<Trackable, Trackable>? _object_map = null;
    private object? _cache = null;
    public TrackableSaver(ObjectGraphView graph_view)
    {
        _graph_view = graph_view;
        
        // TODO: cache when not executing eagerly.
        // including `_cache`, `_file_prefix_feed_tensor`, `_file_prefix_placeholder`,
        // `_object_graph_feed_tensor`, `_object_map`, `_restore_op_cache`, `_saveables_cache`
        
    }

    private (IDictionary<Trackable, IDictionary<string, object>>, IDictionary<Tensor, string>, IDictionary<string, IDictionary<string, Trackable>>, TrackableObjectGraph) 
        gather_serialized_tensors(Tensor? object_graph_tensor = null)
    {
        var (serialized_tensors, feed_additions, registered_savers, graph_proto) = SaveUtil.serialize_graph_view(_graph_view, _object_map, cache:_cache);

        // TODO: cache.

        if(object_graph_tensor is null)
        {
            // tensorflow python: `with ops.device("/cpu:0"):`
            object_graph_tensor = constant_op.constant(graph_proto.ToString(), TF_DataType.TF_STRING);
        }
        else
        {
            feed_additions[object_graph_tensor] = graph_proto.ToString();
        }
        Debug.Assert(!serialized_tensors.ContainsKey(Trackable.None) || !serialized_tensors[Trackable.None].ContainsKey(Trackable.Constants.OBJECT_GRAPH_PROTO_KEY));
        if (serialized_tensors.ContainsKey(Trackable.None))
        {
            serialized_tensors[Trackable.None][Trackable.Constants.OBJECT_GRAPH_PROTO_KEY] = object_graph_tensor;
        }
        return (serialized_tensors, feed_additions, registered_savers, graph_proto);
    }

    private (Tensor, IDictionary<Tensor, string>) save_cached_when_graph_building(Tensor file_prefix, Tensor object_graph_tensor, CheckpointOptions options)
    {
        var (serialized_tensors, feed_additions, registered_savers, graph_proto) = gather_serialized_tensors(object_graph_tensor);

        Func<(Tensor, IDictionary<Tensor, string>)> run_save = () =>
        {
            if (_last_save_object_graph != graph_proto || tf.Context.executing_eagerly() || ops.inside_function())
            {
                var saver = new MultiDeviceSaver(serialized_tensors, registered_savers);
                var save_op = saver.save(file_prefix, options);

                // tensorflow python: `with ops.device("/cpu:0"):`
                using (ops.control_dependencies(new object[] { save_op }))
                {
                    _cached_save_operation = array_ops.identity(file_prefix);
                }
                _last_save_object_graph = graph_proto;
            }
            return (_cached_save_operation, feed_additions);
        };

        if (options.experimental_enable_async_checkpoint)
        {
            throw new NotImplementedException();
        }

        return run_save();
    }

    private (Tensor, IDictionary<Tensor, string>) save_cached_when_graph_building(string file_prefix, Tensor object_graph_tensor, CheckpointOptions options)
    {
        var (serialized_tensors, feed_additions, registered_savers, graph_proto) = gather_serialized_tensors(object_graph_tensor);

        Func<(Tensor, IDictionary<Tensor, string>)> run_save = () =>
        {
            if (_last_save_object_graph != graph_proto || tf.Context.executing_eagerly() || ops.inside_function())
            {
                var saver = new MultiDeviceSaver(serialized_tensors, registered_savers);
                var save_op = saver.save(file_prefix, options);

                // tensorflow python: `with ops.device("/cpu:0"):`
                using (ops.control_dependencies(new object[] {save_op} ))
                {
                    _cached_save_operation = array_ops.identity(tf.constant(file_prefix));
                }
                _last_save_object_graph = graph_proto;
            }
            return (_cached_save_operation, feed_additions);
        };

        if (options.experimental_enable_async_checkpoint)
        {
            throw new NotImplementedException();
        }

        return run_save();
    }

    // TODO: parameter write_done_callback
    public Tensor save(string file_prefix, int? checkpoint_number = null, Session? session = null,
        CheckpointOptions? options = null)
    {
        if (options is null)
        {
            options = new CheckpointOptions();
        }

        Dictionary<Tensor, string> feed_dict = new();
        bool use_session = (!new Context().executing_eagerly() && !ops.inside_function());
        if (checkpoint_number is not null)
        {
            file_prefix = $"{file_prefix}-{checkpoint_number?.ToString()}";
        }

        Tensor file_prefix_tensor;
        Tensor object_graph_tensor;
        if (use_session)
        {
            if (_object_graph_feed_tensor is null)
            {
                // In python there is `with ops.device("/cpu:0")`.
                _object_graph_feed_tensor = constant_op.constant("", TF_DataType.TF_STRING);
                _file_prefix_feed_tensor = constant_op.constant("", TF_DataType.TF_STRING);
            }

            object_graph_tensor = _object_graph_feed_tensor;
            file_prefix_tensor = _file_prefix_feed_tensor;
            feed_dict[file_prefix_tensor] = file_prefix;
        }
        else
        {
            // In python there is `with ops.device("/cpu:0")`.
            file_prefix_tensor = ops.convert_to_tensor(file_prefix, TF_DataType.TF_STRING);
            object_graph_tensor = null;
        }

        var (save_path, new_feed_additions) =
            save_cached_when_graph_building(file_prefix_tensor, object_graph_tensor, options);

        if (new_feed_additions is not null)
        {
            foreach (var pair in new_feed_additions)
            {
                feed_dict.Add(pair.Key, pair.Value);
            }
        }
        if(!use_session)
        {
            session = null;
        }
        else if (session is null)
        {
            session = new Session(); // In python it uses `get_session`.
        }

        if (session is not null)
        {
            var s = feed_dict.Select(x => new FeedItem(x.Key, x.Value)).ToArray();
            return session.run((Tensor)save_path, s);
        }
        else if (use_session)
        {
            throw new RuntimeError($"Unable to save checkpoint to \"{file_prefix}\" " +
                                   "in graph mode without a default session. Please use " +
                                   "`with tf.Session():` to create a session.");
        }
        else
        {
            return save_path;
        }
    }
}