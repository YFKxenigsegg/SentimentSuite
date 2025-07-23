using SentimentSuite.Video.Api.Domain.Videos;
using SentimentSuite.Common.Services;
using System.Security.Cryptography;
using System.Text;
using VideoEntity = SentimentSuite.Video.Api.Domain.Videos.Video;

namespace SentimentSuite.Video.Api.Persistence.Repositories;

public sealed class CachedVideoRepository : IVideoRepository
{
    private readonly IVideoRepository _repository;
    private readonly ICacheService _cacheService;
    private readonly ILogger<CachedVideoRepository> _logger;
    
    // Cache keys
    private const string SUMMARY_KEY_PREFIX = "summary";
    private const string VIDEO_KEY_PREFIX = "video";

    public CachedVideoRepository(
        IVideoRepository repository,
        ICacheService cacheService,
        ILogger<CachedVideoRepository> logger)
    {
        _repository = repository;
        _cacheService = cacheService;
        _logger = logger;
    }

    public async Task<VideoEntity?> GetByUrlAsync(string youtubeUrl, CancellationToken cancellationToken = default)
    {
        var cacheKey = GenerateCacheKey(VIDEO_KEY_PREFIX, youtubeUrl);
        
        // Try cache first
        var cachedVideo = await _cacheService.GetAsync<VideoEntity>(cacheKey, cancellationToken);
        if (cachedVideo != null)
        {
            _logger.LogDebug("Cache hit for video URL: {YoutubeUrl}", youtubeUrl);
            return cachedVideo;
        }

        // Cache miss - query database
        _logger.LogDebug("Cache miss for video URL: {YoutubeUrl}, querying database", youtubeUrl);
        var video = await _repository.GetByUrlAsync(youtubeUrl, cancellationToken);
        
        if (video != null)
        {
            // Cache the result
            await _cacheService.SetAsync(cacheKey, video, cancellationToken: cancellationToken);
            _logger.LogDebug("Cached video for URL: {YoutubeUrl}", youtubeUrl);
        }

        return video;
    }

    public async Task<VideoEntity> CreateAsync(VideoEntity video, CancellationToken cancellationToken = default)
    {
        // Create in database first
        var createdVideo = await _repository.CreateAsync(video, cancellationToken);
        
        // Cache the created video
        var videoCacheKey = GenerateCacheKey(VIDEO_KEY_PREFIX, video.YoutubeUrl);
        await _cacheService.SetAsync(videoCacheKey, createdVideo, cancellationToken: cancellationToken);
        
        // Also cache just the summary for fast retrieval
        var summaryCacheKey = GenerateCacheKey(SUMMARY_KEY_PREFIX, video.YoutubeUrl);
        await _cacheService.SetAsync(summaryCacheKey, new VideoSummaryCache 
        { 
            Summary = createdVideo.Summary,
            CreatedAt = createdVideo.CreatedAt,
            SummaryProvider = createdVideo.SummaryProvider
        }, cancellationToken: cancellationToken);
        
        _logger.LogInformation("Created and cached video: {YoutubeUrl}", video.YoutubeUrl);
        return createdVideo;
    }

    public async Task<IEnumerable<VideoEntity>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        // For GetAll, we typically don't cache due to potential large dataset
        // But we could implement a time-based cache if needed
        return await _repository.GetAllAsync(cancellationToken);
    }

    // Additional method for getting just the summary (faster than full video object)
    public async Task<string?> GetSummaryByUrlAsync(string youtubeUrl, CancellationToken cancellationToken = default)
    {
        var cacheKey = GenerateCacheKey(SUMMARY_KEY_PREFIX, youtubeUrl);
        
        // Try summary cache first
        var cachedSummary = await _cacheService.GetAsync<VideoSummaryCache>(cacheKey, cancellationToken);
        if (cachedSummary != null)
        {
            _logger.LogDebug("Cache hit for summary URL: {YoutubeUrl}", youtubeUrl);
            return cachedSummary.Summary;
        }

        // Fallback to full video lookup
        var video = await GetByUrlAsync(youtubeUrl, cancellationToken);
        return video?.Summary;
    }

    // Cache invalidation method
    public async Task InvalidateCacheAsync(string youtubeUrl, CancellationToken cancellationToken = default)
    {
        var videoCacheKey = GenerateCacheKey(VIDEO_KEY_PREFIX, youtubeUrl);
        var summaryCacheKey = GenerateCacheKey(SUMMARY_KEY_PREFIX, youtubeUrl);
        
        await _cacheService.RemoveAsync(videoCacheKey, cancellationToken);
        await _cacheService.RemoveAsync(summaryCacheKey, cancellationToken);
        
        _logger.LogInformation("Invalidated cache for video: {YoutubeUrl}", youtubeUrl);
    }

    private static string GenerateCacheKey(string prefix, string youtubeUrl)
    {
        // Create a consistent hash of the URL to avoid key length issues
        var urlHash = ComputeSha256Hash(youtubeUrl);
        return $"{prefix}:{urlHash}";
    }

    private static string ComputeSha256Hash(string input)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(bytes).ToLowerInvariant();
    }

    // Lightweight summary cache model
    private sealed class VideoSummaryCache
    {
        public required string Summary { get; init; }
        public DateTime CreatedAt { get; init; }
        public required string SummaryProvider { get; init; }
    }
}
