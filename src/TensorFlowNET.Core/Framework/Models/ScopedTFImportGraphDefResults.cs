using System;

namespace Tensorflow.Framework.Models
{
    public class ScopedTFImportGraphDefResults : ImportGraphDefOptions
    {
        public ScopedTFImportGraphDefResults() : base()
        {
            
        }

        public ScopedTFImportGraphDefResults(IntPtr results) : base(results)
        {

        }

        ~ScopedTFImportGraphDefResults()
        {
            base.Dispose();
        }
    }
}
