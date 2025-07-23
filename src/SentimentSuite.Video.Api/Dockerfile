FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy project files for restore
COPY ["SentimentSuite.Video.Api/SentimentSuite.Video.Api.csproj", "SentimentSuite.Video.Api/"]
COPY ["SentimentSuite.Common/SentimentSuite.Common.csproj", "SentimentSuite.Common/"]

# Restore dependencies
RUN dotnet restore "SentimentSuite.Video.Api/SentimentSuite.Video.Api.csproj"

# Copy source code
COPY ["SentimentSuite.Video.Api/", "SentimentSuite.Video.Api/"]
COPY ["SentimentSuite.Common/", "SentimentSuite.Common/"]

# Build and publish
WORKDIR "/src/SentimentSuite.Video.Api"
RUN dotnet publish "SentimentSuite.Video.Api.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=build /app/publish .
ENTRYPOINT ["dotnet", "SentimentSuite.Video.Api.dll"]