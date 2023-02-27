using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Train;

namespace Tensorflow.Checkpoint;

internal class CheckpointPosition
{
    private CheckpointRestoreCoordinator _checkpoint;
    private int _proto_id;
    private bool _skip_restore;
    public CheckpointPosition(CheckpointRestoreCoordinator checkpoint, int proto_id)
    {
        _checkpoint = checkpoint;
        _proto_id = proto_id;
        _skip_restore = false;
    }

    public Trackable Trackable => _checkpoint.ObjectByProtoId[_proto_id];

    public void restore(Trackable trackable)
    {
        using (ops.init_scope())
        {
            if (bind_project(trackable))
            {

            }
        }
    }

    /// <summary>
    /// Set a checkpoint<->object correspondence.
    /// </summary>
    /// <param name="trackable"></param>
    /// <returns></returns>
    public bool bind_project(Trackable trackable)
    {
        _checkpoint.AllTrackables.Add(trackable);
        _checkpoint.MatchedProtoIds.Add(_proto_id);
        if(_checkpoint.ObjectByProtoId.TryGetValue(_proto_id, out var current_assignment))
        {
            // skip the `logging.warning`.
            return false;
        }
        else
        {
            _checkpoint.ObjectByProtoId[_proto_id] = trackable;
            return true;
        }
    }

    public void gather_ops_or_named_saveables()
    {
        // skip the registered_saver


    }

    /// <summary>
    /// Restore the bound Trackable and dependencies (may be deferred).
    /// </summary>
    private void _restore_descendants()
    {
        Queue<(CheckpointPosition, Trackable)> visit_queue = new();
        visit_queue.Enqueue((this, this.Trackable));

    }

    private void _single_restore()
    {
        var trackable = this.Trackable;
        trackable._maybe_initialize_trackable();
        if(_checkpoint.RestoreUid > trackable.UpdateUid)
        {

        }
    }
}
