using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Threading.Tasks;
using Tensorflow.Hub;

namespace TensorFlowNET.UnitTest.Hub
{
    [TestClass]
    public class MnistModelLoaderTest
    {
        [TestMethod]
        public async Task TestLoad()
        {
            var loader = new MnistModelLoader();
            var result = await loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = "mnist",
                OneHot = true,
                ValidationSize = 5000,
            });

            Assert.IsNotNull(result);
        }
    }
}
