using Tensorflow.Functions;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;
using Tensorflow.Util;

namespace Tensorflow.Keras.Engine;

public interface IModel : ILayer
{
    void compile(IOptimizer optimizer, ILossFunc loss);

    void compile(IOptimizer optimizer, ILossFunc loss, string[] metrics);

    void compile(string optimizer, string loss, string[] metrics);

    void compile(IOptimizer optimizer, ILossFunc loss, IMetricFunc[] metrics);

    ICallback fit(NDArray x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            float validation_split = 0f,
            ValidationDataPack validation_data = null,
            int validation_step = 10,
            bool shuffle = true,
            Dictionary<int, float> class_weight = null,
            NDArray sample_weight = null,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false);

    ICallback fit(IEnumerable<NDArray> x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            float validation_split = 0f,
            ValidationDataPack validation_data = null,
            bool shuffle = true,
            Dictionary<int, float> class_weight = null,
            NDArray sample_weight = null,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false);

    public ICallback fit(IDatasetV2 dataset,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            List<ICallback> callbacks = null,
            IDatasetV2 validation_data = null,
            int validation_step = 10,   // 间隔多少次会进行一次验证
            bool shuffle = true,
            Dictionary<int, float> class_weight = null,
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

    Dictionary<string, float> evaluate(NDArray x, NDArray y,
            int batch_size = -1,
            int verbose = 1,
            NDArray sample_weight = null,

            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false,
            bool return_dict = false,
            bool is_val = false);

    Tensors predict(Tensors x,
            int batch_size = -1,
            int verbose = 0,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false);

    public Tensors predict(IDatasetV2 dataset,
            int batch_size = -1,
            int verbose = 0,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false);

    void summary(int line_length = -1, float[] positions = null);

    IKerasConfig get_config();

    bool Stop_training { get;set; }
}
