namespace Tensorflow
{
    public class BulkSaverBuilder : BaseSaverBuilder, ISaverBuilder
    {
        public BulkSaverBuilder(SaverDef.Types.CheckpointFormatVersion write_version = SaverDef.Types.CheckpointFormatVersion.V2) : base(write_version)
        {

        }
    }
}
