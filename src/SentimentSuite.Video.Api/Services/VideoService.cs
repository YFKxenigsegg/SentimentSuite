using SentimentSuite.Video.Api.Domain.Videos;
using SentimentSuite.Video.Api.Domain.Exceptions;
using SentimentSuite.Video.Api.Persistence.Repositories;
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
    IVideoRepository videoRepository,
    ILogger<VideoService> logger)
    : IVideoService
{
    public async Task<string> GetOrCreateSummaryAsync(string youtubeUrl, CancellationToken cancellationToken = default)
    {
        // Validate YouTube URL format
        if (!IsValidYoutubeUrl(youtubeUrl))
            throw new InvalidYoutubeUrlException(youtubeUrl);

        logger.LogInformation("Processing video summary request for: {YoutubeUrl}", youtubeUrl);

        // Try to get summary from cache first (fastest path)
        if (videoRepository is CachedVideoRepository cachedRepo)
        {
            var cachedSummary = await cachedRepo.GetSummaryByUrlAsync(youtubeUrl, cancellationToken);
            if (!string.IsNullOrEmpty(cachedSummary))
            {
                logger.LogInformation("Cache hit - returning cached summary for: {YoutubeUrl}", youtubeUrl);
                return cachedSummary;
            }
        }
        else
        {
            // Fallback for non-cached repository
            var existing = await videoRepository.GetByUrlAsync(youtubeUrl, cancellationToken);
            if (existing != null)
            {
                logger.LogInformation("Database hit - returning existing summary for: {YoutubeUrl}", youtubeUrl);
                return existing.Summary;
            }
        }

        logger.LogInformation("Cache/database miss - generating new summary for: {YoutubeUrl}", youtubeUrl);

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

            // Persist successful summary (will automatically cache)
            await PersistSummaryAsync(youtubeUrl, transcript, summary, cancellationToken);
            
            logger.LogInformation("Successfully generated and cached summary for: {YoutubeUrl}", youtubeUrl);
            return summary;
        }
        catch (Exception ex) when (ex is not DomainException)
        {
            // Wrap non-domain exceptions
            if (ex is HttpRequestException)
                throw new SummarizationFailedException("External Service", ex);
            
            logger.LogError(ex, "Unexpected error processing video: {YoutubeUrl}", youtubeUrl);
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
            logger.LogDebug("Successfully persisted video summary: {YoutubeUrl}", youtubeUrl);
        }
        catch (Exception ex)
        {
            // Log persistence failure but don't fail the request
            // Return the summary even if we can't persist it
            logger.LogError(ex, "Failed to persist video summary for: {YoutubeUrl}. Summary will still be returned.", youtubeUrl);
            
            // Could implement retry logic or dead letter queue here
            // For now, we gracefully continue
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
