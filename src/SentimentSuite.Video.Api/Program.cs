using SentimentSuite.Video.Api.Models;
using SentimentSuite.Video.Api.Services;
using YoutubeExplode;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddJsonFile("secrets.json", optional: true);
builder.Services.Configure<AnthropicOptions>(builder.Configuration.GetSection("Anthropic"));
builder.Services.Configure<LocalSummaryOptions>(builder.Configuration.GetSection("LocalSummary"));
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton<YoutubeClient>();
builder.Services.AddScoped<YoutubeTranscriptService>();
builder.Services.AddScoped<IPromptService, PromptService>();
builder.Services.AddHttpClient<LocalSummaryService>();
builder.Services.AddHttpClient<SonnetSummaryService>();
builder.Services.AddScoped<ITextSummaryService, HybridSummaryService>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapControllers();

app.Run();
