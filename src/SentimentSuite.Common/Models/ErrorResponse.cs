namespace SentimentSuite.Common.Models;

public sealed class ErrorResponse
{
    public required string Type { get; init; }
    public required string Title { get; init; }
    public required int Status { get; init; }
    public required string Detail { get; init; }
    public string? TraceId { get; init; }
    public Dictionary<string, object>? Extensions { get; init; }

    public static ErrorResponse Create(string type, string title, int status, string detail, string? traceId = null, Dictionary<string, object>? extensions = null)
    {
        return new ErrorResponse
        {
            Type = type,
            Title = title,
            Status = status,
            Detail = detail,
            TraceId = traceId,
            Extensions = extensions
        };
    }
} 