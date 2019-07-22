﻿namespace Tensorflow.Framework.Models
{
    public class ScopedTFImportGraphDefOptions : ImportGraphDefOptions
    {
        public ScopedTFImportGraphDefOptions() : base()
        {

        }

        ~ScopedTFImportGraphDefOptions()
        {
            base.Dispose();
        }
    }
}
