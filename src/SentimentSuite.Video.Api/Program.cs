using MongoDB.Driver;
using SentimentSuite.Video.Api.Models;
using SentimentSuite.Video.Api.Domain.Videos;
using SentimentSuite.Video.Api.Persistence.Options;
using SentimentSuite.Video.Api.Persistence.Repositories;
using SentimentSuite.Video.Api.Services;
using SentimentSuite.Common.Extensions;
using YoutubeExplode;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddJsonFile("secrets.json", optional: true);

// Configure options
builder.Services.Configure<MongoDbOptions>(builder.Configuration.GetSection("MongoDB"));
builder.Services.Configure<AnthropicOptions>(builder.Configuration.GetSection("Anthropic"));
builder.Services.Configure<LocalSummaryOptions>(builder.Configuration.GetSection("LocalSummary"));

// Add common services
builder.Services.AddSentimentSuiteCommon();

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
builder.Services.AddScoped<IVideoRepository, VideoRepository>();

// HTTP clients and summary services
builder.Services.AddHttpClient<LocalSummaryService>();
builder.Services.AddHttpClient<SonnetSummaryService>();
builder.Services.AddScoped<ITextSummaryService, HybridSummaryService>();

var app = builder.Build();

// Exception handling middleware (must be first)
app.UseSentimentSuiteExceptionHandling();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapControllers();

app.Run();
