using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace Tensorflow.Hub
{
    public interface IModelLoader<TDataSet>
        where TDataSet : IDataSet
    {
        Task<Datasets<TDataSet>> LoadAsync(ModelLoadSetting setting);
    }
}
