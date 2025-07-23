using SentimentSuite.Video.Api.Models;
using SentimentSuite.Video.Api.Domain.Exceptions;

namespace SentimentSuite.Video.Api.Services;

public sealed class SonnetSummaryService(
    HttpClient httpClient,
    IOptions<AnthropicOptions> options,
    IPromptService promptService,
    ILogger<SonnetSummaryService> logger)
    : ITextSummaryService
{
    private readonly HttpClient _httpClient = httpClient;
    private readonly AnthropicOptions _opts = options.Value;
    private readonly IPromptService _promptService = promptService;

    public async Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken)
    {
        try
        {
            logger.LogInformation("Sending request to Anthropic Claude API");
            
            var request = CreateRequest(transcript);
            var httpRequest = CreateHttpRequest(request);

            using var response = await _httpClient.SendAsync(httpRequest, cancellationToken);
            
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                logger.LogError("Anthropic API returned {StatusCode}: {ErrorContent}", 
                    response.StatusCode, errorContent);
                
                throw new SummarizationFailedException("Anthropic Claude", 
                    $"HTTP {response.StatusCode}: {response.ReasonPhrase}");
            }

            var anthropicResponse = await response.Content.ReadFromJsonAsync<AnthropicResponse>(cancellationToken);
            var summary = anthropicResponse?.Content.FirstOrDefault()?.Text;
            
            if (string.IsNullOrWhiteSpace(summary))
            {
                logger.LogWarning("Anthropic API returned empty summary");
                throw new SummarizationFailedException("Anthropic Claude", "Empty summary returned");
            }

            logger.LogInformation("Successfully received summary from Anthropic Claude");
            return summary;
        }
        catch (HttpRequestException ex)
        {
            logger.LogError(ex, "Network error communicating with Anthropic API");
            throw new SummarizationFailedException("Anthropic Claude", ex);
        }
        catch (TaskCanceledException ex) when (ex.CancellationToken.IsCancellationRequested)
        {
            logger.LogWarning("Request to Anthropic API was cancelled");
            throw;
        }
        catch (Exception ex) when (ex is not SummarizationFailedException)
        {
            logger.LogError(ex, "Unexpected error in Anthropic summarization service");
            throw new SummarizationFailedException("Anthropic Claude", ex);
        }
    }

    private AnthropicRequest CreateRequest(string transcript) =>
        new(_opts.Model, 1024, [
            new("user", _promptService.CreateSummaryPrompt(transcript))
        ]);

    private HttpRequestMessage CreateHttpRequest(AnthropicRequest request)
    {
        var httpRequest = new HttpRequestMessage(HttpMethod.Post, _opts.ApiUrl)
        {
            Content = JsonContent.Create(request)
        };

        httpRequest.Headers.Add("x-api-key", _opts.ApiKey);
        httpRequest.Headers.Add("anthropic-version", "2023-06-01");

        return httpRequest;
    }
}
