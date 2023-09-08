using Tensorflow.Common.Types;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;
using Tensorflow.Training;

namespace Tensorflow.Keras
{
    public interface ILayer: IWithTrackable, IKerasConfigable
    {
        string Name { get; }
        bool Trainable { get; }
        bool Built { get; }
        void build(KerasShapesWrapper input_shape);
        List<ILayer> Layers { get; }
        List<INode> InboundNodes { get; }
        List<INode> OutboundNodes { get; }
        Tensors Apply(Tensors inputs, Tensors states = null, bool? training = false, IOptionalArgs? optional_args = null);
        List<IVariableV1> TrainableVariables { get; }
        List<IVariableV1> TrainableWeights { get; }
        List<IVariableV1> NonTrainableWeights { get; }
        List<IVariableV1> Weights { get; set; }
        void set_weights(IEnumerable<NDArray> weights);
        List<NDArray> get_weights();
        Shape OutputShape { get; }
        KerasShapesWrapper BatchInputShape { get; }
        KerasShapesWrapper BuildInputShape { get; }
        TF_DataType DType { get; }
        int count_params();
        void adapt(Tensor data, int? batch_size = null, int? steps = null);
    }
}
