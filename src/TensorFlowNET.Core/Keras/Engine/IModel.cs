using Tensorflow.Functions;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.Engine;

public interface IModel : ILayer
{
    void compile(IOptimizer optimizer = null,
            ILossFunc loss = null,
            string[] metrics = null);

    void compile(string optimizer, string loss, string[] metrics);

    ICallback fit(NDArray x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            float validation_split = 0f,
            bool shuffle = true,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false);

    ICallback fit(IEnumerable<NDArray> x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            float validation_split = 0f,
            bool shuffle = true,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false);

    void save(string filepath,
            bool overwrite = true,
            bool include_optimizer = true,
            string save_format = "tf",
            SaveOptions? options = null,
            ConcreteFunction? signatures = null,
            bool save_traces = true);

    void save_weights(string filepath, 
        bool overwrite = true, 
        string save_format = null, 
        object options = null);

    void load_weights(string filepath, 
        bool by_name = false, 
        bool skip_mismatch = false, 
        object options = null);

    void evaluate(NDArray x, NDArray y,
            int batch_size = -1,
            int verbose = 1,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false,
            bool return_dict = false);

    Tensors predict(Tensor x,
            int batch_size = -1,
            int verbose = 0,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false);

    void summary(int line_length = -1, float[] positions = null);

    IKerasConfig get_config();
}
