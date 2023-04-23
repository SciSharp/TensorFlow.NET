using System;
using System.Net;

namespace Tensorflow.Hub
{
    public class HttpUncompressedFileResolver : HttpResolverBase
    {
        private readonly PathResolver _pathResolver;

        public HttpUncompressedFileResolver()
        {
            _pathResolver = new PathResolver();
        }

        public override string Call(string handle)
        {
            handle = AppendUncompressedFormatQuery(handle);
            var gsLocation = RequestGcsLocation(handle);
            return _pathResolver.Call(gsLocation);
        }

        public override bool IsSupported(string handle)
        {
            if (!is_http_protocol(handle))
            {
                return false;
            }

            var load_format = resolver.model_load_format();
            return load_format == Enum.GetName(typeof(resolver.ModelLoadFormat), resolver.ModelLoadFormat.UNCOMPRESSED);
        }

        protected virtual string AppendUncompressedFormatQuery(string handle)
        {
            return append_format_query(handle, ("tf-hub-format", "uncompressed"));
        }

        protected virtual string RequestGcsLocation(string handleWithParams)
        {
            var request = WebRequest.Create(handleWithParams);
            var response = request.GetResponse() as HttpWebResponse;

            if (response == null)
            {
                throw new Exception("Failed to get a response from the server.");
            }

            var statusCode = (int)response.StatusCode;

            if (statusCode != 303)
            {
                throw new Exception($"Expected 303 for GCS location lookup but got HTTP {statusCode} {response.StatusDescription}");
            }

            var location = response.Headers["Location"];

            if (!location.StartsWith("gs://"))
            {
                throw new Exception($"Expected Location:GS path but received {location}");
            }

            return location;
        }
    }
}