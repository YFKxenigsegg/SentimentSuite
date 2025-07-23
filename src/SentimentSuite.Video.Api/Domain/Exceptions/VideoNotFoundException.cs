using SentimentSuite.Common.Exceptions;

namespace SentimentSuite.Video.Api.Domain.Exceptions;

public sealed class VideoNotFoundException : ResourceNotFoundException
{
    public VideoNotFoundException(string youtubeUrl) 
        : base("Video", youtubeUrl, $"Video not found for URL: {youtubeUrl}")
    {
    }
} 