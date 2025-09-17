# ai-speaker-detection
Identify different speakers on an audio.

On this process, the Speaker Detection updates the information allocated in the transcription provided.


## Test Lambda

```bash
# Build image
docker build -t speaker:v1 .

# Run lambda locally
docker run -p 9000:8080 speaker:v1 .
```

Send a payload:
```bash
curl -X GET "http://localhost:8080/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
      "httpMethod": "GET",
      "path": "/health",
      "headers": {},
      "body": null
    }'
```

## Test locally on docker

Execute on local docker, providing a file to process in S3:  TODO
```bash
docker run --rm \
  -e AWS_SERVICE_BUCKET=ai-scriber \
  speaker:v1 \
  '{"job_id": "test", "s3_key": "test-file.mp3"}'
```
