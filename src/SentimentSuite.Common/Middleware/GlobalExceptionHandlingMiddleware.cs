using System.Diagnostics;
using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using SentimentSuite.Common.Exceptions;
using SentimentSuite.Common.Models;

namespace SentimentSuite.Common.Middleware;

public sealed class GlobalExceptionHandlingMiddleware(RequestDelegate next, ILogger<GlobalExceptionHandlingMiddleware> logger)
{
    private readonly RequestDelegate _next = next;
    private readonly ILogger<GlobalExceptionHandlingMiddleware> _logger = logger;

    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await _next(context);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "An unhandled exception occurred while processing the request");
            await HandleExceptionAsync(context, ex);
        }
    }

    private async Task HandleExceptionAsync(HttpContext context, Exception exception)
    {
        var response = MapExceptionToErrorResponse(exception, context);
        
        context.Response.ContentType = "application/json";
        context.Response.StatusCode = response.Status;

        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true
        };

        var jsonResponse = JsonSerializer.Serialize(response, jsonOptions);
        await context.Response.WriteAsync(jsonResponse);
    }

    private ErrorResponse MapExceptionToErrorResponse(Exception exception, HttpContext context)
    {
        var traceId = Activity.Current?.Id ?? context.TraceIdentifier;

        return exception switch
        {
            ValidationException ex => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.5.1",
                title: "Validation Error",
                status: (int)HttpStatusCode.BadRequest,
                detail: ex.Message,
                traceId: traceId,
                extensions: new Dictionary<string, object> 
                { 
                    ["field"] = ex.Field,
                    ["value"] = ex.Value ?? "null"
                }
            ),

            ResourceNotFoundException ex => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.5.4",
                title: "Resource Not Found",
                status: (int)HttpStatusCode.NotFound,
                detail: ex.Message,
                traceId: traceId,
                extensions: new Dictionary<string, object> 
                { 
                    ["resourceType"] = ex.ResourceType,
                    ["resourceId"] = ex.ResourceId
                }
            ),

            ExternalServiceException ex => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.6.3",
                title: "External Service Error",
                status: ex.StatusCode ?? (int)HttpStatusCode.BadGateway,
                detail: ex.Message,
                traceId: traceId,
                extensions: new Dictionary<string, object> { ["serviceName"] = ex.ServiceName }
            ),

            DomainException ex => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.5.1",
                title: "Domain Error",
                status: (int)HttpStatusCode.BadRequest,
                detail: ex.Message,
                traceId: traceId
            ),

            // Handle YoutubeExplode exceptions that slip through
            Exception ex when ex.GetType().Namespace?.StartsWith("YoutubeExplode.Exceptions") == true => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.6.3",
                title: "YouTube Service Error",
                status: (int)HttpStatusCode.BadGateway,
                detail: "An error occurred while accessing YouTube services",
                traceId: traceId,
                extensions: new Dictionary<string, object> { ["serviceName"] = "YouTube API" }
            ),

            HttpRequestException ex => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.6.3",
                title: "External Service Error",
                status: (int)HttpStatusCode.BadGateway,
                detail: "An error occurred while communicating with an external service",
                traceId: traceId
            ),

            TaskCanceledException ex when ex.CancellationToken.IsCancellationRequested => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.5.8",
                title: "Request Timeout",
                status: (int)HttpStatusCode.RequestTimeout,
                detail: "The request was cancelled due to timeout",
                traceId: traceId
            ),

            OperationCanceledException => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.5.8",
                title: "Request Cancelled",
                status: (int)HttpStatusCode.RequestTimeout,
                detail: "The request was cancelled",
                traceId: traceId
            ),

            ArgumentException ex => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.5.1",
                title: "Invalid Argument",
                status: (int)HttpStatusCode.BadRequest,
                detail: ex.Message,
                traceId: traceId
            ),

            _ => ErrorResponse.Create(
                type: "https://tools.ietf.org/html/rfc7231#section-6.6.1",
                title: "Internal Server Error",
                status: (int)HttpStatusCode.InternalServerError,
                detail: "An unexpected error occurred while processing your request", 
                traceId: traceId
            )
        };
    }
} 