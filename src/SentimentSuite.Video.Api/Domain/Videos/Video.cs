using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace SentimentSuite.Video.Api.Domain.Videos;

public sealed class Video
{
    [BsonId]
    public ObjectId Id { get; set; }
    
    [BsonElement("youtubeUrl")]
    public required string YoutubeUrl { get; init; }
    
    [BsonElement("videoId")]
    public required string VideoId { get; init; }
    
    [BsonElement("title")]
    public required string Title { get; init; }
    
    [BsonElement("transcript")]
    public required string Transcript { get; init; }
    
    [BsonElement("summary")]
    public required string Summary { get; init; }
    
    [BsonElement("createdAt")]
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    
    [BsonElement("summaryProvider")]
    public required string SummaryProvider { get; init; }
}