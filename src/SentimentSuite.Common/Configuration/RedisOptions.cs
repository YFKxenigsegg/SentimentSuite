namespace SentimentSuite.Common.Configuration;

public sealed class RedisOptions
{
    public required string ConnectionString { get; init; }
    public int DatabaseId { get; init; } = 0;
    public TimeSpan DefaultExpiration { get; init; } = TimeSpan.FromHours(24);
    public string KeyPrefix { get; init; } = "SentimentSuite:";
}
