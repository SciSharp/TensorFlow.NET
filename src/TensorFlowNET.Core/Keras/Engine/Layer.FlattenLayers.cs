using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        public IEnumerable<Layer> _flatten_layers(bool recursive = true, bool include_self = true)
        {
            if (include_self)
                yield return this;

            var seen_object_ids = new List<int>();
            var deque = new Queue<Layer>(_layers);
            while (!deque.empty())
            {
                var layer_or_container = deque.Dequeue();
                var layer_or_container_id = layer_or_container.GetHashCode();
                if (seen_object_ids.Contains(layer_or_container_id))
                    continue;
                seen_object_ids.Add(layer_or_container_id);
                yield return layer_or_container;
                if (recursive)
                    deque.extendleft(layer_or_container._layers);
            }
        }
    }
}
