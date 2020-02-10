using Keras.Layers;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    public class TensorFlowOpLayer : Layer
    {
        public TensorFlowOpLayer(string node_def, string name, NDArray[] constants = null, bool trainable = true, string dtype = null)
        {

        }

        public override void call(Tensor[] inputs)
        {
            throw new NotImplementedException();
        }

        public override Dictionary<string, object> get_config()
        {
            throw new NotImplementedException();
        }

        private NodeDef _make_node_def(Graph graph) => throw new NotImplementedException();

        private Tensor[] _make_op(Tensor[] inputs) => throw new NotImplementedException();

        private Tensor[] _defun_call(Tensor[] inputs) => throw new NotImplementedException();
    }

    public class AddLoss : Layer
    {
        public AddLoss(bool unconditional)
        {
            throw new NotImplementedException();
        }

        public override void call(Tensor[] inputs)
        {
            throw new NotImplementedException();
        }

        public override Dictionary<string, object> get_config()
        {
            throw new NotImplementedException();
        }
    }

    public class AddMetric : Layer
    {
        public AddMetric(string aggregation = null, string metric_name = null)
        {
            throw new NotImplementedException();
        }

        public override void call(Tensor[] inputs)
        {
            throw new NotImplementedException();
        }

        public override Dictionary<string, object> get_config()
        {
            throw new NotImplementedException();
        }
    }

    public class KerasHistory
    {

    }
}
