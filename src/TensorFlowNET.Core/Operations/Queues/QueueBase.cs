using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Queues
{
    public class QueueBase
    {
        protected TF_DataType[] _dtypes;
        protected TensorShape[] _shapes;
        protected string[] _names;
        protected Tensor _queue_ref;
        protected string _name;

        public QueueBase(TF_DataType[] dtypes, TensorShape[] shapes, string[] names)
        {
            _dtypes = dtypes;
            _shapes = shapes;
            _names = names;
        }

        public Operation enqueue(Tensor val, string name = null)
        {
            return tf_with(ops.name_scope(name, $"{_name}_enqueue", val), scope =>
            {
                var vals = new[] { val };
                if (_queue_ref.dtype == TF_DataType.TF_RESOURCE)
                    return gen_data_flow_ops.queue_enqueue_v2(_queue_ref, vals, name: scope);
                else
                    return gen_data_flow_ops.queue_enqueue(_queue_ref, vals, name: scope);
            });
        }

        public Tensor[] dequeue_many(int n, string name = null)
        {
            if (name == null)
                name = $"{_name}_DequeueMany";

            var ret = gen_data_flow_ops.queue_dequeue_many_v2(_queue_ref, n: n, component_types: _dtypes, name: name);
            //var op = ret[0].op;
            //var cv = tensor_util.constant_value(op.inputs[1]);
            //var batch_dim = new Dimension(cv);

            return _dequeue_return_value(ret);
        }

        public Tensor[] _dequeue_return_value(Tensor[] tensors)
        {
            if (_names != null)
                throw new NotImplementedException("");
            return tensors;
        }
    }
}
