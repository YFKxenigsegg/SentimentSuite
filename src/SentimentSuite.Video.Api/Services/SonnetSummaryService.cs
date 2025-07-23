using SentimentSuite.Video.Api.Models;

namespace SentimentSuite.Video.Api.Services;

public sealed class SonnetSummaryService(
    HttpClient httpClient,
    IOptions<AnthropicOptions> options,
    IPromptService promptService)
    : ITextSummaryService
{
    private readonly HttpClient _httpClient = httpClient;
    private readonly AnthropicOptions _opts = options.Value;
    private readonly IPromptService _promptService = promptService;

    public async Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken)
    {
        var request = CreateRequest(transcript);
        var httpRequest = CreateHttpRequest(request);

        using var response = await _httpClient.SendAsync(httpRequest, cancellationToken);
        response.EnsureSuccessStatusCode();

        var anthropicResponse = await response.Content.ReadFromJsonAsync<AnthropicResponse>(cancellationToken);
        return anthropicResponse?.Content.FirstOrDefault()?.Text ?? string.Empty;
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
