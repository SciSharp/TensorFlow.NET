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
        Tensors Apply(Tensors inputs, Tensor state = null, bool training = false);
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

        Tensors Call(Tensors inputs, Tensor? mask = null, bool? training = null, Tensors? initial_state = null, Tensors? constants = null);

        StateSizeWrapper state_size { get; }

        int output_size { get; }
    }
}
