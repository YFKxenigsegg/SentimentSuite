namespace SentimentSuite.Common.Exceptions;

public class ExternalServiceException : DomainException
{
    public string ServiceName { get; }
    public int? StatusCode { get; }

    public ExternalServiceException(string serviceName, string message) 
        : base($"External service '{serviceName}' error: {message}")
    {
        ServiceName = serviceName;
    }

    public ExternalServiceException(string serviceName, int statusCode, string message) 
        : base($"External service '{serviceName}' returned {statusCode}: {message}")
    {
        ServiceName = serviceName;
        StatusCode = statusCode;
    }

    public ExternalServiceException(string serviceName, Exception innerException) 
        : base($"External service '{serviceName}' error: {innerException.Message}", innerException)
    {
        ServiceName = serviceName;
    }
} 