using SentimentSuite.Common.Exceptions;

namespace SentimentSuite.Video.Api.Domain.Exceptions;

public sealed class SummarizationFailedException : ExternalServiceException
{
    public SummarizationFailedException(string provider, string message) 
        : base($"Summarization Service ({provider})", message)
    {
    }

    public SummarizationFailedException(string provider, Exception innerException) 
        : base($"Summarization Service ({provider})", innerException)
    {
    }

    public SummarizationFailedException(string provider, int statusCode, string message) 
        : base($"Summarization Service ({provider})", statusCode, message)
    {
    }
} 