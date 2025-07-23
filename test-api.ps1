# SentimentSuite API Testing Script for Windows PowerShell
# Run this after starting docker-compose up -d

Write-Host "🚀 Testing SentimentSuite API with Redis Caching" -ForegroundColor Green
Write-Host "=" * 50

# Test API health endpoint
Write-Host "`n1. Testing API Health Endpoint..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:5156/health"
    Write-Host "✅ API Health Status: $($healthResponse.status)" -ForegroundColor Green
    
    # Check individual health checks if available
    if ($healthResponse.entries) {
        foreach ($entry in $healthResponse.entries.PSObject.Properties) {
            $status = $entry.Value.status
            $color = if ($status -eq "Healthy") { "Green" } else { "Red" }
            Write-Host "  - $($entry.Name): $status" -ForegroundColor $color
        }
    }
} catch {
    Write-Host "❌ API Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Make sure containers are running: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}

# Test API availability
Write-Host "`n2. Testing API Swagger..." -ForegroundColor Yellow
try {
    $healthCheck = Invoke-WebRequest -Uri "http://localhost:5156/swagger/index.html" -UseBasicParsing
    Write-Host "✅ API Swagger is accessible on http://localhost:5156/swagger" -ForegroundColor Green
} catch {
    Write-Host "❌ API Swagger is not accessible" -ForegroundColor Red
}

# Test Redis health directly
Write-Host "`n3. Testing Redis Health..." -ForegroundColor Yellow
try {
    $redisTest = docker exec sentimentsuite-redis-1 redis-cli ping
    if ($redisTest -eq "PONG") {
        Write-Host "✅ Redis is running and healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Redis health check failed" -ForegroundColor Red
}

# Test Python Summarizer
Write-Host "`n4. Testing Python Summarizer..." -ForegroundColor Yellow
try {
    $summarizerHealth = Invoke-RestMethod -Uri "http://localhost:8000/health"
    if ($summarizerHealth.status -eq "healthy") {
        Write-Host "✅ Python Summarizer is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Python Summarizer health check failed" -ForegroundColor Red
}

# Clear any existing cache for clean test
Write-Host "`n5. Clearing Redis Cache for Clean Test..." -ForegroundColor Yellow
try {
    docker exec sentimentsuite-redis-1 redis-cli FLUSHDB
    Write-Host "✅ Redis cache cleared" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not clear Redis cache" -ForegroundColor Yellow
}

# Prepare test data
$testUrl = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
$requestBody = @{
    youtubeUrl = $testUrl
} | ConvertTo-Json

Write-Host "`n6. Testing Cache Miss (First Request)..." -ForegroundColor Yellow
Write-Host "This will take 5-15 seconds to generate summary..." -ForegroundColor Cyan

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $firstResponse = Invoke-RestMethod -Uri "http://localhost:5156/api/summary" `
        -Method POST `
        -Body $requestBody `
        -ContentType "application/json"
    
    $stopwatch.Stop()
    $firstTime = $stopwatch.ElapsedMilliseconds
    
    Write-Host "✅ First request completed in: $firstTime ms" -ForegroundColor Green
    Write-Host "Summary preview: $($firstResponse.summary.Substring(0, [Math]::Min(100, $firstResponse.summary.Length)))..." -ForegroundColor White
} catch {
    Write-Host "❌ First request failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Error details: $($_.ErrorDetails.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n7. Testing Cache Hit (Second Request)..." -ForegroundColor Yellow
Write-Host "This should be much faster (cached)..." -ForegroundColor Cyan

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $secondResponse = Invoke-RestMethod -Uri "http://localhost:5156/api/summary" `
        -Method POST `
        -Body $requestBody `
        -ContentType "application/json"
    
    $stopwatch.Stop()
    $secondTime = $stopwatch.ElapsedMilliseconds
    
    Write-Host "✅ Second request completed in: $secondTime ms" -ForegroundColor Green
    
    # Calculate performance improvement
    if ($secondTime -gt 0) {
        $improvement = [Math]::Round(($firstTime / $secondTime), 2)
        Write-Host "🚀 Performance improvement: ${improvement}x faster!" -ForegroundColor Magenta
    } else {
        Write-Host "🚀 Performance improvement: Extremely fast (< 1ms)!" -ForegroundColor Magenta
    }
    
    # Verify same content
    if ($firstResponse.summary -eq $secondResponse.summary) {
        Write-Host "✅ Cache returned identical content" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Cache returned different content" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Second request failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n8. Checking Redis Cache..." -ForegroundColor Yellow
try {
    $cacheKeys = docker exec sentimentsuite-redis-1 redis-cli KEYS "SentimentSuite:Video:*"
    if ($cacheKeys) {
        $keyCount = ($cacheKeys | Measure-Object).Count
        Write-Host "✅ Found $keyCount cache keys in Redis" -ForegroundColor Green
        
        if ($keyCount -gt 0) {
            Write-Host "Cache keys:" -ForegroundColor Cyan
            $cacheKeys | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
        }
    } else {
        Write-Host "⚠️  No cache keys found in Redis" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Could not check Redis cache" -ForegroundColor Red
}

# Final health check
Write-Host "`n9. Final System Health Check..." -ForegroundColor Yellow
try {
    $finalHealth = Invoke-RestMethod -Uri "http://localhost:5156/health"
    Write-Host "✅ System Health: $($finalHealth.status)" -ForegroundColor Green
} catch {
    Write-Host "❌ System health check failed" -ForegroundColor Red
}

Write-Host "`n10. Performance Summary" -ForegroundColor Yellow
Write-Host "=" * 30
Write-Host "First Request (Cache Miss): $firstTime ms" -ForegroundColor White
Write-Host "Second Request (Cache Hit): $secondTime ms" -ForegroundColor White
if ($secondTime -gt 0) {
    $improvement = [Math]::Round(($firstTime / $secondTime), 2)
    Write-Host "Performance Gain: ${improvement}x faster" -ForegroundColor Green
} else {
    Write-Host "Performance Gain: Extremely fast!" -ForegroundColor Green
}

if ($secondTime -lt 500) {
    Write-Host "🎉 Excellent! Cache is working perfectly!" -ForegroundColor Green
} elseif ($secondTime -lt 2000) {
    Write-Host "✅ Good! Cache is working well!" -ForegroundColor Yellow
} else {
    Write-Host "⚠️  Cache might not be working optimally" -ForegroundColor Red
}

Write-Host "`n🎯 Testing Complete!" -ForegroundColor Green
Write-Host "📊 Visit http://localhost:5156/health for system health" -ForegroundColor Cyan
Write-Host "📚 Visit http://localhost:5156/swagger to test manually" -ForegroundColor Cyan 