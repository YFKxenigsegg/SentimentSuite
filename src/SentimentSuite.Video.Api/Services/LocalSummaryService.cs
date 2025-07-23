using SentimentSuite.Video.Api.Models;

namespace SentimentSuite.Video.Api.Services;

public sealed class LocalSummaryService(
    HttpClient httpClient,
    IOptions<LocalSummaryOptions> options)
    : ITextSummaryService
{
    private readonly HttpClient _httpClient = httpClient;
    private readonly string _baseUrl = options.Value.BaseUrl;

    public async Task<string> SummarizeAsync(string transcript, CancellationToken cancellationToken)
    {
        var response = await _httpClient.PostAsJsonAsync(
            $"{_baseUrl.TrimEnd('/')}/summarize",
            new { text = transcript },
            cancellationToken);

        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<SummaryResponse>(cancellationToken);
        return result?.Summary ?? string.Empty;
    }

    private class SummaryResponse
    {
        public string Summary { get; set; } = default!;
    }
}
