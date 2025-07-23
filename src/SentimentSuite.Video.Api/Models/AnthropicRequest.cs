namespace SentimentSuite.Video.Api.Models;

public sealed record AnthropicRequest(
    string Model,
    int MaxTokens,
    AnthropicMessage[] Messages);

public sealed record AnthropicMessage(string Role, string Content);

public sealed record AnthropicResponse(AnthropicContent[] Content);
public sealed record AnthropicContent(string Text);
