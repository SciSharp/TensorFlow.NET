using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using System.IO;
using NumSharp;

namespace Tensorflow.Hub
{
    public class MnistModelLoader : IModelLoader<MnistDataSet>
    {
        private const string DEFAULT_SOURCE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
        private const string TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
        private const string TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
        private const string TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
        private const string TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

        public static async Task<Datasets<MnistDataSet>> LoadAsync(string trainDir, bool oneHot = false)
        {
            var loader = new MnistModelLoader();
            return await loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = trainDir,
                OneHot = oneHot
            });
        }

        public async Task<Datasets<MnistDataSet>> LoadAsync(ModelLoadSetting setting)
        {
            if (setting.TrainSize.HasValue && setting.ValidationSize >= setting.TrainSize.Value)
                throw new ArgumentException("Validation set should be smaller than training set");

            var sourceUrl = setting.SourceUrl;

            if (string.IsNullOrEmpty(sourceUrl))
                sourceUrl = DEFAULT_SOURCE_URL;

            // load train images
            await this.DownloadAsync(sourceUrl + TRAIN_IMAGES, setting.TrainDir, TRAIN_IMAGES)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            await this.UnzipAsync(Path.Combine(setting.TrainDir, TRAIN_IMAGES), setting.TrainDir)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            var trainImages = ExtractImages(Path.Combine(setting.TrainDir, Path.GetFileNameWithoutExtension(TRAIN_IMAGES)), limit: setting.TrainSize);

            // load train labels
            await this.DownloadAsync(sourceUrl + TRAIN_LABELS, setting.TrainDir, TRAIN_LABELS)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            await this.UnzipAsync(Path.Combine(setting.TrainDir, TRAIN_LABELS), setting.TrainDir)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            var trainLabels = ExtractLabels(Path.Combine(setting.TrainDir, Path.GetFileNameWithoutExtension(TRAIN_LABELS)), one_hot: setting.OneHot, limit: setting.TrainSize);

            // load test images
            await this.DownloadAsync(sourceUrl + TEST_IMAGES, setting.TrainDir, TEST_IMAGES)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            await this.UnzipAsync(Path.Combine(setting.TrainDir, TEST_IMAGES), setting.TrainDir)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            var testImages = ExtractImages(Path.Combine(setting.TrainDir, Path.GetFileNameWithoutExtension(TEST_IMAGES)), limit: setting.TestSize);

            // load test labels
            await this.DownloadAsync(sourceUrl + TEST_LABELS, setting.TrainDir, TEST_LABELS)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            await this.UnzipAsync(Path.Combine(setting.TrainDir, TEST_LABELS), setting.TrainDir)
                .ShowProgressInConsole(setting.ShowProgressInConsole);

            var testLabels = ExtractLabels(Path.Combine(setting.TrainDir, Path.GetFileNameWithoutExtension(TEST_LABELS)), one_hot: setting.OneHot, limit: setting.TestSize);

            var end = trainImages.shape[0];

            var validationSize = setting.ValidationSize;

            var validationImages = trainImages[np.arange(validationSize)];
            var validationLabels = trainLabels[np.arange(validationSize)];
            
            trainImages = trainImages[np.arange(validationSize, end)];
            trainLabels = trainLabels[np.arange(validationSize, end)];

            var dtype = setting.DataType;
            var reshape = setting.ReShape;

            var train = new MnistDataSet(trainImages, trainLabels, dtype, reshape);
            var validation = new MnistDataSet(validationImages, validationLabels, dtype, reshape);
            var test = new MnistDataSet(trainImages, trainLabels, dtype, reshape);

            return new Datasets<MnistDataSet>(train, validation, test);
        }

        private NDArray ExtractImages(string file, int? limit = null)
        {
            if (!Path.IsPathRooted(file))
                file = Path.Combine(AppContext.BaseDirectory, file);

            using (var bytestream = new FileStream(file, FileMode.Open))
            {
                var magic = Read32(bytestream);
                if (magic != 2051)
                    throw new Exception($"Invalid magic number {magic} in MNIST image file: {file}");
                
                var num_images =  Read32(bytestream);
                num_images = limit == null ? num_images : Math.Min(num_images, (uint)limit);

                var rows = Read32(bytestream);
                var cols = Read32(bytestream);

                var buf = new byte[rows * cols * num_images];

                bytestream.Read(buf, 0, buf.Length);

                var data = np.frombuffer(buf, np.uint8);
                data = data.reshape((int)num_images, (int)rows, (int)cols, 1);

                return data;
            }
        }

        private NDArray ExtractLabels(string file, bool one_hot = false, int num_classes = 10, int? limit = null)
        {
            if (!Path.IsPathRooted(file))
                file = Path.Combine(AppContext.BaseDirectory, file);
                
            using (var bytestream = new FileStream(file, FileMode.Open))
            {
                var magic = Read32(bytestream);
                if (magic != 2049)
                    throw new Exception($"Invalid magic number {magic} in MNIST label file: {file}");
                
                var num_items = Read32(bytestream);
                num_items = limit == null ? num_items : Math.Min(num_items, (uint)limit);
                
                var buf = new byte[num_items];

                bytestream.Read(buf, 0, buf.Length);
                
                var labels = np.frombuffer(buf, np.uint8);

                if (one_hot)
                    return DenseToOneHot(labels, num_classes);
                
                return labels;
            }
        }

        private NDArray DenseToOneHot(NDArray labels_dense, int num_classes)
        {
            var num_labels = labels_dense.shape[0];
            var index_offset = np.arange(num_labels) * num_classes;
            var labels_one_hot = np.zeros(num_labels, num_classes);
            var labels = labels_dense.Data<byte>();
            for (int row = 0; row < num_labels; row++)
            {
                var col = labels[row];
                labels_one_hot.SetData(1.0, row, col);
            }

            return labels_one_hot;
        }

        private uint Read32(FileStream bytestream)
        {
            var buffer = new byte[sizeof(uint)];
            var count = bytestream.Read(buffer, 0, 4);
            return np.frombuffer(buffer, ">u4").Data<uint>()[0];
        }
    }
}
