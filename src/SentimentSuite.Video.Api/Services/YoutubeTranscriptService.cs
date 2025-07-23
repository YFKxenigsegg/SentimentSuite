using YoutubeExplode;
using YoutubeExplode.Videos;
using YoutubeExplode.Exceptions;
using SentimentSuite.Common.Exceptions;
using SentimentSuite.Video.Api.Domain.Exceptions;

namespace SentimentSuite.Video.Api.Services;

public sealed class YoutubeTranscriptService(
    YoutubeClient youtubeClient,
    ILogger<YoutubeTranscriptService> logger)
{
    private readonly YoutubeClient _youtubeClient = youtubeClient;

    public async Task<string> GetTranscriptAsync(string youtubeUrl, CancellationToken cancellationToken)
    {
        try
        {
            logger.LogInformation("Retrieving transcript for YouTube URL: {YoutubeUrl}", youtubeUrl);
            
            var videoId = VideoId.Parse(youtubeUrl);
            var tracks = await _youtubeClient.Videos.ClosedCaptions.GetManifestAsync(videoId, cancellationToken);
            
            var trackInfo = tracks.GetByLanguage("en") ?? tracks.Tracks.FirstOrDefault();
            if (trackInfo is null)
            {
                logger.LogWarning("No captions found for video: {YoutubeUrl}", youtubeUrl);
                throw new TranscriptNotFoundException(youtubeUrl);
            }
            
            var track = await _youtubeClient.Videos.ClosedCaptions.GetAsync(trackInfo, cancellationToken);
            var transcript = string.Join(" ", track.Captions.Select(c => c.Text));
            
            if (string.IsNullOrWhiteSpace(transcript))
            {
                logger.LogWarning("Empty transcript retrieved for video: {YoutubeUrl}", youtubeUrl);
                throw new TranscriptNotFoundException(youtubeUrl);
            }
            
            logger.LogInformation("Successfully retrieved transcript for video: {YoutubeUrl}, Length: {Length} characters", 
                youtubeUrl, transcript.Length);
            
            return transcript;
        }
        catch (YoutubeExplode.Exceptions.VideoUnavailableException ex)
        {
            logger.LogError(ex, "Video is unavailable: {YoutubeUrl}", youtubeUrl);
            throw new Domain.Exceptions.VideoUnavailableException(youtubeUrl, ex.Message);
        }
        catch (VideoUnplayableException ex)
        {
            logger.LogError(ex, "Video is unplayable: {YoutubeUrl}", youtubeUrl);
            throw new Domain.Exceptions.VideoUnavailableException(youtubeUrl, "Video is unplayable");
        }
        catch (RequestLimitExceededException ex)
        {
            logger.LogError(ex, "YouTube API request limit exceeded for: {YoutubeUrl}", youtubeUrl);
            throw new SummarizationFailedException("YouTube API", "Request limit exceeded. Please try again later.");
        }
        catch (YoutubeExplodeException ex)
        {
            logger.LogError(ex, "YouTube API error for: {YoutubeUrl}", youtubeUrl);
            throw new SummarizationFailedException("YouTube API", ex.Message);
        }
        catch (ArgumentException ex) when (ex.Message.Contains("Invalid YouTube video ID"))
        {
            logger.LogError(ex, "Invalid YouTube URL format: {YoutubeUrl}", youtubeUrl);
            throw new InvalidYoutubeUrlException(youtubeUrl, ex);
        }
        catch (HttpRequestException ex)
        {
            logger.LogError(ex, "Network error while accessing YouTube for: {YoutubeUrl}", youtubeUrl);
            throw new SummarizationFailedException("YouTube API", ex);
        }
        catch (Exception ex) when (ex is not DomainException)
        {
            logger.LogError(ex, "Unexpected error retrieving transcript for: {YoutubeUrl}", youtubeUrl);
            throw new TranscriptNotFoundException(youtubeUrl);
        }
    }
}
