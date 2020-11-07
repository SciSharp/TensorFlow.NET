using System.Threading.Tasks;

namespace Tensorflow
{
    public interface IModelLoader<TDataSet>
        where TDataSet : IDataSet
    {
        Task<Datasets<TDataSet>> LoadAsync(ModelLoadSetting setting);
    }
}
