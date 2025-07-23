using YoutubeExplode;
using YoutubeExplode.Videos;

namespace SentimentSuite.Video.Api.Services;

public sealed class YoutubeTranscriptService(YoutubeClient youtubeClient)
{
    private readonly YoutubeClient _youtubeClient = youtubeClient;

    public async Task<string> GetTranscriptAsync(string youtubeUrl, CancellationToken cancellationToken)
    {
        var videoId = VideoId.Parse(youtubeUrl);
        var tracks = await _youtubeClient.Videos.ClosedCaptions.GetManifestAsync(videoId, cancellationToken);
        var trackInfo = tracks.GetByLanguage("en") ?? tracks.Tracks.FirstOrDefault();
        if (trackInfo is null)
        {
            return string.Empty;
        }
        var track = await _youtubeClient.Videos.ClosedCaptions.GetAsync(trackInfo, cancellationToken);
        return string.Join(" ", track.Captions.Select(c => c.Text));
    }
}
