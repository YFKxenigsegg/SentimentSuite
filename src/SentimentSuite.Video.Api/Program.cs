using MongoDB.Driver;
using SentimentSuite.Video.Api.Models;
using SentimentSuite.Video.Api.Domain.Videos;
using SentimentSuite.Video.Api.Persistence.Options;
using SentimentSuite.Video.Api.Persistence.Repositories;
using SentimentSuite.Video.Api.Services;
using SentimentSuite.Common.Extensions;
using SentimentSuite.Common.Configuration;
using YoutubeExplode;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddJsonFile("secrets.json", optional: true);

// Configure options
builder.Services.Configure<MongoDbOptions>(builder.Configuration.GetSection("MongoDB"));
builder.Services.Configure<AnthropicOptions>(builder.Configuration.GetSection("Anthropic"));
builder.Services.Configure<LocalSummaryOptions>(builder.Configuration.GetSection("LocalSummary"));
builder.Services.Configure<RedisOptions>(builder.Configuration.GetSection("Redis"));

// Add common services
builder.Services.AddSentimentSuiteCommon();

// Add Redis caching
builder.Services.AddSentimentSuiteRedis();

// Add health checks
builder.Services.AddHealthChecks()
    .AddCheck("self", () => Microsoft.Extensions.Diagnostics.HealthChecks.HealthCheckResult.Healthy())
    .AddRedis(builder.Configuration.GetSection("Redis")["ConnectionString"] ?? "localhost:6379", 
        name: "redis", 
        tags: new[] { "redis", "cache" })
    .AddMongoDb(builder.Configuration.GetSection("MongoDB")["ConnectionString"] ?? "mongodb://localhost:27017",
        name: "mongodb",
        tags: new[] { "mongodb", "database" });

// MongoDB client
builder.Services.AddSingleton<IMongoClient>(sp =>
{
    var options = sp.GetRequiredService<IOptions<MongoDbOptions>>().Value;
    return new MongoClient(options.ConnectionString);
});

// ASP.NET Core services
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// External dependencies
builder.Services.AddSingleton<YoutubeClient>();

// Application services
builder.Services.AddScoped<YoutubeTranscriptService>();
builder.Services.AddScoped<IPromptService, PromptService>();
builder.Services.AddScoped<IVideoService, VideoService>();

// Repository pattern with caching
builder.Services.AddScoped<VideoRepository>(); // Base repository implementation
builder.Services.AddScoped<IVideoRepository>(serviceProvider =>
{
    var baseRepository = serviceProvider.GetRequiredService<VideoRepository>();
    var cacheService = serviceProvider.GetRequiredService<SentimentSuite.Common.Services.ICacheService>();
    var logger = serviceProvider.GetRequiredService<ILogger<CachedVideoRepository>>();
    return new CachedVideoRepository(baseRepository, cacheService, logger);
});

// HTTP clients and summary services
builder.Services.AddHttpClient<LocalSummaryService>();
builder.Services.AddHttpClient<SonnetSummaryService>();
builder.Services.AddScoped<ITextSummaryService, HybridSummaryService>();

var app = builder.Build();

// Exception handling middleware (must be first)
app.UseSentimentSuiteExceptionHandling();

// Health checks endpoint
app.MapHealthChecks("/health");

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapControllers();

app.Run();
