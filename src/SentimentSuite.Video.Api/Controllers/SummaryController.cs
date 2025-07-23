using Microsoft.AspNetCore.Mvc;
using SentimentSuite.Video.Api.Models;
using SentimentSuite.Video.Api.Services;

namespace SentimentSuite.Video.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class SummaryController(
    YoutubeTranscriptService transcriptService,
    ITextSummaryService summaryService)
    : ControllerBase
{
    [HttpPost]
    [Route("")]
    public async Task<ActionResult<SummaryResponse>> Post(
        [FromBody] SummaryRequest request,
        CancellationToken cancellationToken)
    {
        var transcript = await transcriptService.GetTranscriptAsync(request.YoutubeUrl, cancellationToken);
        if (string.IsNullOrWhiteSpace(transcript))
        {
            return BadRequest();
        }

        var summary = await summaryService.SummarizeAsync(transcript, cancellationToken);
        return Ok(new SummaryResponse { Summary = summary });
    }
}
