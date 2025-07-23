namespace SentimentSuite.Video.Api.Models;

public sealed class AnthropicOptions
{
    public required string ApiKey { get; init; }
    public required string Model { get; init; }
    public required string ApiUrl { get; init; }
}
