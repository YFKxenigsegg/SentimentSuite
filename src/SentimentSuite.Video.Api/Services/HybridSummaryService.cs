using SentimentSuite.Video.Api.Domain.Exceptions;

namespace SentimentSuite.Video.Api.Services;

public interface IHybridSummaryService
{
    Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken);
}

public sealed class HybridSummaryService(
    SonnetSummaryService sonnet,
    LocalSummaryService local,
    ILogger<HybridSummaryService> logger)
    : ITextSummaryService
{
    public async Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken)
    {
        // Try Anthropic Claude first
        try
        {
            logger.LogInformation("Attempting summarization with Anthropic Claude");
            var result = await sonnet.SummarizeAsync(transcript, cancellationToken);
            logger.LogInformation("Successfully summarized using Anthropic Claude");
            return result;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Anthropic Claude summarization failed, falling back to local service");
        }

        // Fallback to local service
        try
        {
            logger.LogInformation("Attempting summarization with local service");
            var result = await local.SummarizeAsync(transcript, cancellationToken);
            logger.LogInformation("Successfully summarized using local service");
            return result;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Local summarization service failed");
            throw new SummarizationFailedException("Local Service", ex);
        }
    }
}
