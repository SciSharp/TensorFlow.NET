using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Training;

namespace Tensorflow.Keras
{
    public interface ILayer: IWithTrackable, IKerasConfigable
    {
        string Name { get; }
        bool Trainable { get; }
        bool Built { get; }
        void build(Shape input_shape);
        List<ILayer> Layers { get; }
        List<INode> InboundNodes { get; }
        List<INode> OutboundNodes { get; }
        Tensors Apply(Tensors inputs, Tensor state = null, bool is_training = false);
        List<IVariableV1> TrainableVariables { get; }
        List<IVariableV1> TrainableWeights { get; }
        List<IVariableV1> NonTrainableWeights { get; }
        Shape OutputShape { get; }
        Shape BatchInputShape { get; }
        TensorShapeConfig BuildInputShape { get; }
        TF_DataType DType { get; }
        int count_params();
    }
}
