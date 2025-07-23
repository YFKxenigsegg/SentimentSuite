namespace SentimentSuite.Video.Api.Services;

public sealed class PromptService : IPromptService
{
    public string CreateSummaryPrompt(string transcript) =>
        $"Summarize the following transcript into 3-5 paragraphs: {transcript}";
}
