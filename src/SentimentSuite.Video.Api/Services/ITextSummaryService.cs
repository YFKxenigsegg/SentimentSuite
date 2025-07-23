namespace SentimentSuite.Video.Api.Services;

public interface ITextSummaryService
{
    Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken);
}
