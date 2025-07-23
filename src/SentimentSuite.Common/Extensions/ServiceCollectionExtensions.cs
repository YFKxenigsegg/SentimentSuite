using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using SentimentSuite.Common.Middleware;

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