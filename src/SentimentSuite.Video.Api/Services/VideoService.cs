using SentimentSuite.Video.Api.Domain.Videos;
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
        // Check cache first
        var existing = await videoRepository.GetByUrlAsync(youtubeUrl, cancellationToken);
        if (existing != null)
            return existing.Summary;

        // Get transcript
        var transcript = await transcriptService.GetTranscriptAsync(youtubeUrl, cancellationToken);
        if (string.IsNullOrWhiteSpace(transcript))
            throw new InvalidOperationException("Could not retrieve transcript");

        // Generate summary
        var summary = await summaryService.SummarizeAsync(transcript, cancellationToken);

        // Persist only if successful
        if (!string.IsNullOrWhiteSpace(summary))
        {
            await PersistSummaryAsync(youtubeUrl, transcript, summary, cancellationToken);
        }

        return summary;
    }

    private async Task PersistSummaryAsync(string youtubeUrl, string transcript, string summary, CancellationToken cancellationToken)
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
}
