using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace Tensorflow.Hub
{
    public class MnistModelLoader : IModelLoader<MnistDataSet>
    {
        public Task<Datasets<MnistDataSet>> LoadAsync(ModelLoadSetting setting)
        {
            throw new NotImplementedException();
        }
    }
}
