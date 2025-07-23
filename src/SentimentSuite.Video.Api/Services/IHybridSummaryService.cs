namespace SentimentSuite.Video.Api.Services;

public interface IHybridSummaryService
{
    Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken);
}
