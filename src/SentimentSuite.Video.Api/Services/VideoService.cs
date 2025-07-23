using SentimentSuite.Video.Api.Domain.Videos;
using SentimentSuite.Video.Api.Domain.Exceptions;
using SentimentSuite.Common.Exceptions;
using YoutubeExplode.Videos;

namespace SentimentSuite.Video.Api.Services;

public interface IVideoService
{
    Task<string> GetOrCreateSummaryAsync(string youtubeUrl, CancellationToken cancellationToken = default);
}

public sealed class VideoService(
    YoutubeTranscriptService transcriptService,
    ITextSummaryService summaryService,
    IVideoRepository videoRepository)
    : IVideoService
{
    public async Task<string> GetOrCreateSummaryAsync(string youtubeUrl, CancellationToken cancellationToken = default)
    {
        // Validate YouTube URL format
        if (!IsValidYoutubeUrl(youtubeUrl))
            throw new InvalidYoutubeUrlException(youtubeUrl);

        // Check cache first
        var existing = await videoRepository.GetByUrlAsync(youtubeUrl, cancellationToken);
        if (existing != null)
            return existing.Summary;

        try
        {
            // Get transcript
            var transcript = await transcriptService.GetTranscriptAsync(youtubeUrl, cancellationToken);
            if (string.IsNullOrWhiteSpace(transcript))
                throw new TranscriptNotFoundException(youtubeUrl);

            // Generate summary
            var summary = await summaryService.SummarizeAsync(transcript, cancellationToken);
            if (string.IsNullOrWhiteSpace(summary))
                throw new SummarizationFailedException("Unknown", "Summary generation returned empty result");

            // Persist successful summary
            await PersistSummaryAsync(youtubeUrl, transcript, summary, cancellationToken);
            return summary;
        }
        catch (Exception ex) when (ex is not DomainException)
        {
            // Wrap non-domain exceptions
            if (ex is HttpRequestException)
                throw new SummarizationFailedException("External Service", ex);
            
            throw;
        }
    }

    private async Task PersistSummaryAsync(string youtubeUrl, string transcript, string summary, CancellationToken cancellationToken)
    {
        try
        {
            var videoId = VideoId.Parse(youtubeUrl).Value;
            var video = new Domain.Videos.Video
            {
                YoutubeUrl = youtubeUrl,
                VideoId = videoId,
                Title = $"Video {videoId}",
                Transcript = transcript,
                Summary = summary,
                SummaryProvider = summaryService.GetType().Name
            };

            await videoRepository.CreateAsync(video, cancellationToken);
        }
        catch (Exception ex)
        {
            // Log persistence failure but don't fail the request
            // Return the summary even if we can't persist it
            // TODO: Consider implementing retry logic or dead letter queue
        }
    }

    private static bool IsValidYoutubeUrl(string url)
    {
        if (string.IsNullOrWhiteSpace(url))
            return false;

        try
        {
            var videoId = VideoId.TryParse(url);
            return videoId.HasValue;
        }
        catch
        {
            return false;
        }
    }
}
