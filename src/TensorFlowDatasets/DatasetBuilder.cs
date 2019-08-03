using System;

namespace TensorFlowDatasets
{
    /// <summary>
    /// Abstract base class for all datasets.
    /// </summary>
    public class DatasetBuilder
    {
        /// <summary>
        /// Downloads and prepares dataset for reading.
        /// </summary>
        /// <param name="download_dir">
        /// directory where downloaded files are stored.
        /// </param>
        /// <param name="download_config">
        /// further configuration for downloading and preparing dataset.
        /// </param>
        public void download_and_prepare(string download_dir = null, DownloadConfig download_config = null)
        {

        }
    }
}
