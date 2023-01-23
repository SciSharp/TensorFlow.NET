using System;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Train;
using static Tensorflow.ApiDef.Types;
using static Tensorflow.CostGraphDef.Types;
using static Tensorflow.OptimizerOptions.Types;

namespace Tensorflow.Checkpoint
{
    /// <summary>
    /// Saves checkpoints directly from multiple devices.
    /// Note that this is a low-level utility which stores Tensors in the keys
    /// specified by `SaveableObject`s.Higher-level utilities for object-based
    /// checkpointing are built on top of it.
    /// </summary>
    public class MultiDeviceSaver
    {
        public MultiDeviceSaver(IDictionary<Trackable, IDictionary<string, object>> serialized_tensors, 
            IDictionary<string, IDictionary<string, Trackable>>? registered_savers = null, bool call_with_mapped_capture = false)
        {

        }

        public Operation? save(string file_prefix, CheckpointOptions? options= null)
        {
            throw new NotImplementedException();
        }

        public Operation? save(Tensor file_prefix, CheckpointOptions? options = null)
        {
            throw new NotImplementedException();
        }
    }
}
