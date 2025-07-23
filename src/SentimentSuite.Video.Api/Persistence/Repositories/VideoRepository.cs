using MongoDB.Driver;
using MongoDB.Driver.Linq;
using SentimentSuite.Video.Api.Persistence.Options;
using VideoEntity = SentimentSuite.Video.Api.Domain.Videos.Video;
using SentimentSuite.Video.Api.Domain.Videos;

namespace SentimentSuite.Video.Api.Persistence.Repositories;

public sealed class VideoRepository : IVideoRepository
{
    private readonly IMongoCollection<VideoEntity> _collection;

    public VideoRepository(IMongoClient mongoClient, IOptions<MongoDbOptions> options)
    {
        var database = mongoClient.GetDatabase(options.Value.DatabaseName);
        _collection = database.GetCollection<VideoEntity>("videos");
        
        // Create index on YoutubeUrl for faster lookups
        var indexKeys = Builders<VideoEntity>.IndexKeys.Ascending(x => x.YoutubeUrl);
        var indexOptions = new CreateIndexOptions { Unique = true };
        _collection.Indexes.CreateOneAsync(new CreateIndexModel<VideoEntity>(indexKeys, indexOptions));
    }

    public async Task<VideoEntity?> GetByUrlAsync(string youtubeUrl, CancellationToken cancellationToken = default)
    {
        var filter = Builders<VideoEntity>.Filter.Eq(x => x.YoutubeUrl, youtubeUrl);
        return await _collection.Find(filter).FirstOrDefaultAsync(cancellationToken);
    }

    public async Task<VideoEntity> CreateAsync(VideoEntity video, CancellationToken cancellationToken = default)
    {
        await _collection.InsertOneAsync(video, cancellationToken: cancellationToken);
        return video;
    }

    public async Task<IEnumerable<VideoEntity>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        return await _collection.Find(Builders<VideoEntity>.Filter.Empty)
            .SortByDescending(x => x.CreatedAt)
            .ToListAsync(cancellationToken);
    }
}
