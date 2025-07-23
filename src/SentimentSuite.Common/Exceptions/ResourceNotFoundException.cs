namespace SentimentSuite.Common.Exceptions;

public class ResourceNotFoundException : DomainException
{
    public string ResourceType { get; }
    public string ResourceId { get; }

    public ResourceNotFoundException(string resourceType, string resourceId) 
        : base($"{resourceType} not found with ID: {resourceId}")
    {
        ResourceType = resourceType;
        ResourceId = resourceId;
    }

    public ResourceNotFoundException(string resourceType, string resourceId, string customMessage) 
        : base(customMessage)
    {
        ResourceType = resourceType;
        ResourceId = resourceId;
    }
} 