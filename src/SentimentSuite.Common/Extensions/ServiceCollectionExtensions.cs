using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Logging;
using SentimentSuite.Common.Middleware;
using SentimentSuite.Common.Configuration;
using SentimentSuite.Common.Services;
using StackExchange.Redis;
using System.Linq;

namespace SentimentSuite.Common.Extensions;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds SentimentSuite common services to the service collection
    /// </summary>
    public static IServiceCollection AddSentimentSuiteCommon(this IServiceCollection services)
    {
        // Add any common services here in the future
        // For example: logging interceptors, correlation ID services, etc.
        
        return services;
    }

    /// <summary>
    /// Adds Redis caching services to the service collection
    /// </summary>
    public static IServiceCollection AddSentimentSuiteRedis(this IServiceCollection services)
    {
        services.AddSingleton<IConnectionMultiplexer>(serviceProvider =>
        {
            var options = serviceProvider.GetRequiredService<IOptions<RedisOptions>>().Value;
            var logger = serviceProvider.GetRequiredService<ILogger<IConnectionMultiplexer>>();
            
            var configuration = ConfigurationOptions.Parse(options.ConnectionString);
            configuration.AbortOnConnectFail = false; // Don't fail on initial connection issues
            configuration.ConnectRetry = 3;
            configuration.ConnectTimeout = 5000;
            configuration.SyncTimeout = 5000;
            
            var connection = ConnectionMultiplexer.Connect(configuration);
            
            // Log connection events
            connection.ConnectionFailed += (sender, args) =>
            {
                logger.LogError("Redis connection failed: {EndPoint} - {FailureType}", 
                    args.EndPoint, args.FailureType);
            };
            
            connection.ConnectionRestored += (sender, args) =>
            {
                logger.LogInformation("Redis connection restored: {EndPoint}", args.EndPoint);
            };
            
            connection.ErrorMessage += (sender, args) =>
            {
                logger.LogError("Redis error: {Message}", args.Message);
            };
            
            logger.LogInformation("Redis connection established: {EndPoints}", 
                string.Join(", ", connection.GetEndPoints().Select(ep => ep.ToString())));
            
            return connection;
        });

        services.AddScoped<ICacheService, RedisCacheService>();
        
        return services;
    }
}

public static class ApplicationBuilderExtensions
{
    /// <summary>
    /// Adds global exception handling middleware to the application pipeline
    /// </summary>
    public static IApplicationBuilder UseSentimentSuiteExceptionHandling(this IApplicationBuilder app)
    {
        return app.UseMiddleware<GlobalExceptionHandlingMiddleware>();
    }
}
