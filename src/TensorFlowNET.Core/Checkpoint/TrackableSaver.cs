using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Contexts;
using Tensorflow.Eager;

namespace Tensorflow.Checkpoint;

public class TrackableSaver
{
    private ObjectGraphView _graph_view;
    private EagerTensor _cached_save_operation;
    private TrackableObjectGraph _last_save_object_graph;
    private Tensor? _object_graph_feed_tensor = null;
    private Tensor? _file_prefix_feed_tensor = null;
    public TrackableSaver(ObjectGraphView graph_view)
    {
        _graph_view = graph_view;
        
        // TODO: cache when not executing eagerly.
        // including `_cache`, `_file_prefix_feed_tensor`, `_file_prefix_placeholder`,
        // `_object_graph_feed_tensor`, `_object_map`, `_restore_op_cache`, `_saveables_cache`
        
    }

    private void gather_serialized_tensors(Tensor? object_graph_tensor = null)
    {
        throw new NotImplementedException();
    }

    private (EagerTensor, IDictionary<Tensor, string>) save_cached_when_graph_building(Tensor file_prefix, Tensor object_graph_tensor, CheckpointOptions options)
    {
        throw new NotImplementedException();
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
                _object_graph_feed_tensor = constant_op.constant("", dtypes.variant);
                _file_prefix_feed_tensor = constant_op.constant("", dtypes.variant);
            }

            object_graph_tensor = _object_graph_feed_tensor;
            file_prefix_tensor = _file_prefix_feed_tensor;
            feed_dict[file_prefix_tensor] = file_prefix;
        }
        else
        {
            // In python there is `with ops.device("/cpu:0")`.
            file_prefix_tensor = ops.convert_to_tensor(file_prefix, dtypes.variant);
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