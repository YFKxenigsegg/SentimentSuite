using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SentimentSuite.Common.Configuration;
using StackExchange.Redis;

namespace SentimentSuite.Common.Services;

public sealed class RedisCacheService : ICacheService
{
    private readonly IDatabase _database;
    private readonly IConnectionMultiplexer _connectionMultiplexer;
    private readonly RedisOptions _options;
    private readonly ILogger<RedisCacheService> _logger;
    private readonly JsonSerializerOptions _jsonOptions;

    public RedisCacheService(
        IConnectionMultiplexer connectionMultiplexer,
        IOptions<RedisOptions> options,
        ILogger<RedisCacheService> logger)
    {
        _connectionMultiplexer = connectionMultiplexer;
        _options = options.Value;
        _logger = logger;
        _database = _connectionMultiplexer.GetDatabase(_options.DatabaseId);
        
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false
        };
    }

    public async Task<T?> GetAsync<T>(string key, CancellationToken cancellationToken = default) where T : class
    {
        try
        {
            var fullKey = BuildKey(key);
            var value = await _database.StringGetAsync(fullKey);
            
            if (!value.HasValue)
            {
                _logger.LogDebug("Cache miss for key: {Key}", fullKey);
                return null;
            }

            _logger.LogDebug("Cache hit for key: {Key}", fullKey);
            return JsonSerializer.Deserialize<T>(value!, _jsonOptions);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting cache value for key: {Key}", key);
            return null; // Fail gracefully, don't break the application
        }
    }

    public async Task SetAsync<T>(string key, T value, TimeSpan? expiration = null, CancellationToken cancellationToken = default) where T : class
    {
        try
        {
            var fullKey = BuildKey(key);
            var jsonValue = JsonSerializer.Serialize(value, _jsonOptions);
            var expirationTime = expiration ?? _options.DefaultExpiration;
            
            await _database.StringSetAsync(fullKey, jsonValue, expirationTime);
            _logger.LogDebug("Cached value for key: {Key} with expiration: {Expiration}", fullKey, expirationTime);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting cache value for key: {Key}", key);
            // Don't throw - cache failures shouldn't break the application
        }
    }

    public async Task RemoveAsync(string key, CancellationToken cancellationToken = default)
    {
        try
        {
            var fullKey = BuildKey(key);
            await _database.KeyDeleteAsync(fullKey);
            _logger.LogDebug("Removed cache key: {Key}", fullKey);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing cache key: {Key}", key);
        }
    }

    public async Task RemoveByPatternAsync(string pattern, CancellationToken cancellationToken = default)
    {
        try
        {
            var server = _connectionMultiplexer.GetServer(_connectionMultiplexer.GetEndPoints().First());
            var fullPattern = BuildKey(pattern);
            
            await foreach (var key in server.KeysAsync(pattern: fullPattern))
            {
                await _database.KeyDeleteAsync(key);
            }
            
            _logger.LogDebug("Removed cache keys matching pattern: {Pattern}", fullPattern);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing cache keys by pattern: {Pattern}", pattern);
        }
    }

    public async Task<bool> ExistsAsync(string key, CancellationToken cancellationToken = default)
    {
        try
        {
            var fullKey = BuildKey(key);
            return await _database.KeyExistsAsync(fullKey);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking cache key existence: {Key}", key);
            return false;
        }
    }

    public string BuildKey(params string[] keyParts)
    {
        return _options.KeyPrefix + string.Join(":", keyParts);
    }
}
