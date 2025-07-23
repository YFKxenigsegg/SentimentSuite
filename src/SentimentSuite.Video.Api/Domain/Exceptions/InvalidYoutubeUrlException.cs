using SentimentSuite.Common.Exceptions;

namespace SentimentSuite.Video.Api.Domain.Exceptions;

public sealed class InvalidYoutubeUrlException : ValidationException
{
    public InvalidYoutubeUrlException(string providedUrl) 
        : base("youtubeUrl", providedUrl, $"Invalid YouTube URL format: {providedUrl}")
    {
    }

    public InvalidYoutubeUrlException(string providedUrl, Exception innerException) 
        : base("youtubeUrl", providedUrl, $"Invalid YouTube URL format: {providedUrl}", innerException)
    {
    }
} 