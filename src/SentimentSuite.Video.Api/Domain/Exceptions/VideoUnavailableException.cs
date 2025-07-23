using SentimentSuite.Common.Exceptions;

namespace SentimentSuite.Video.Api.Domain.Exceptions;

public sealed class VideoUnavailableException : ResourceNotFoundException
{
    public VideoUnavailableException(string youtubeUrl, string reason) 
        : base("Video", youtubeUrl, $"Video is unavailable: {reason}")
    {
    }

    public VideoUnavailableException(string youtubeUrl) 
        : base("Video", youtubeUrl, "Video is unavailable")
    {
    }
} 