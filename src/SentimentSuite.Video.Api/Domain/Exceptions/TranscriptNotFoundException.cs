using SentimentSuite.Common.Exceptions;

namespace SentimentSuite.Video.Api.Domain.Exceptions;

public sealed class TranscriptNotFoundException : ResourceNotFoundException
{
    public TranscriptNotFoundException(string youtubeUrl) 
        : base("Transcript", youtubeUrl, $"No transcript available for video: {youtubeUrl}")
    {
    }
} 