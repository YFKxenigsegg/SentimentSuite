using SentimentSuite.Video.Api.Models;
using SentimentSuite.Video.Api.Services;

namespace SentimentSuite.Video.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class SummaryController(IVideoService videoService) : ControllerBase
{
    [HttpPost]
    [Route("")]
    public async Task<ActionResult<SummaryResponse>> Post(
        [FromBody] SummaryRequest request,
        CancellationToken cancellationToken)
    {
        var summary = await videoService.GetOrCreateSummaryAsync(request.YoutubeUrl, cancellationToken);
        return Ok(new SummaryResponse { Summary = summary });
    }
}
