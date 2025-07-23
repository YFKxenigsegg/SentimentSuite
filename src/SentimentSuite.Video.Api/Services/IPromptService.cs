namespace SentimentSuite.Video.Api.Services;

public interface IPromptService
{
    string CreateSummaryPrompt(string transcript);
}
