using SentimentSuite.Video.Api.Models;
using SentimentSuite.Video.Api.Domain.Exceptions;

namespace SentimentSuite.Video.Api.Services;

public sealed class LocalSummaryService(
    HttpClient httpClient,
    IOptions<LocalSummaryOptions> options,
    ILogger<LocalSummaryService> logger)
    : ITextSummaryService
{
    private readonly HttpClient _httpClient = httpClient;
    private readonly string _baseUrl = options.Value.BaseUrl;

    public async Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken)
    {
        try
        {
            logger.LogInformation("Sending request to local summarization service at {BaseUrl}", _baseUrl);
            
            var response = await _httpClient.PostAsJsonAsync(
                $"{_baseUrl.TrimEnd('/')}/summarize",
                new { text = transcript },
                cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                logger.LogError("Local summarization service returned {StatusCode}: {ErrorContent}", 
                    response.StatusCode, errorContent);
                
                throw new SummarizationFailedException("Local Service", 
                    $"HTTP {response.StatusCode}: {response.ReasonPhrase}");
            }

            var result = await response.Content.ReadFromJsonAsync<SummaryResponse>(cancellationToken);
            
            if (result?.Summary is null or "")
            {
                logger.LogWarning("Local summarization service returned empty summary");
                throw new SummarizationFailedException("Local Service", "Empty summary returned");
            }

            logger.LogInformation("Successfully received summary from local service");
            return result.Summary;
        }
        catch (HttpRequestException ex)
        {
            logger.LogError(ex, "Network error communicating with local summarization service");
            throw new SummarizationFailedException("Local Service", ex);
        }
        catch (TaskCanceledException ex) when (ex.CancellationToken.IsCancellationRequested)
        {
            logger.LogWarning("Request to local summarization service was cancelled");
            throw;
        }
        catch (Exception ex) when (ex is not SummarizationFailedException)
        {
            logger.LogError(ex, "Unexpected error in local summarization service");
            throw new SummarizationFailedException("Local Service", ex);
        }
    }

    private sealed class SummaryResponse
    {
        public string Summary { get; set; } = default!;
    }
}
