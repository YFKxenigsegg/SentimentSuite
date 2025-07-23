namespace SentimentSuite.Video.Api.Services;

public sealed class HybridSummaryService(
    SonnetSummaryService sonnet,
    LocalSummaryService local)
    : ITextSummaryService
{
    public async Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken)
    {
        try
        {
            return await sonnet.SummarizeAsync(transcript, cancellationToken);
        }
        catch
        {
            return await local.SummarizeAsync(transcript, cancellationToken);
        }
    }
}
