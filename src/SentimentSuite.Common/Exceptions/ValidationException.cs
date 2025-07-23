namespace SentimentSuite.Common.Exceptions;

public class ValidationException : DomainException
{
    public string Field { get; }
    public object? Value { get; }

    public ValidationException(string field, object? value, string message) 
        : base(message)
    {
        Field = field;
        Value = value;
    }

    public ValidationException(string field, object? value, string message, Exception innerException) 
        : base(message, innerException)
    {
        Field = field;
        Value = value;
    }
} 