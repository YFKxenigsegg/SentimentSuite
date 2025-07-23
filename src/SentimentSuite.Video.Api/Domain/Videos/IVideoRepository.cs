namespace SentimentSuite.Video.Api.Domain.Videos;

public interface IVideoRepository
{
    Task<Video?> GetByUrlAsync(string youtubeUrl, CancellationToken cancellationToken = default);
    Task<Video> CreateAsync(Video video, CancellationToken cancellationToken = default);
    Task<IEnumerable<Video>> GetAllAsync(CancellationToken cancellationToken = default);
}